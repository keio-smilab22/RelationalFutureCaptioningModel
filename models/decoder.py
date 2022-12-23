import torch
from torch import nn
import torch.nn.functional as F

from models.attentions import MultiHeadAttention
from models.misc import FeedforwardNeuralNetModel, make_pad_shifted_mask


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.attention = MultiHeadAttention(cfg)
        self.output = TrmFeedForward(cfg)
        self.ffn = FeedforwardNeuralNetModel(cfg.hidden_size, cfg.hidden_size * 2, cfg.hidden_size)
        self.rand = torch.randn(1, requires_grad=True).cuda()
        self.rand_z = torch.randn(1, requires_grad=True).cuda()
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, x, attention_mask, clip_his, make_knn_dstore: bool=False):
        """
        Args:
            prev_m: (N, M, D)
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        Returns:
        """
        # attention layer
        identity_x = x.clone().cuda()
        x = self.attention(x=x, source_kv=clip_his)
        x = self.rand * identity_x + (1 - self.rand) * x
        x = self.LayerNorm(x)

        if make_knn_dstore:
            knn_feat = x.clone().detach().cpu()
        
        # ffn layer
        identity_x = x.clone().cuda()
        x = self.ffn(x)
        x = self.rand_z * identity_x + (1 - self.rand_z) * x
        x = self.LayerNorm(x)

        if make_knn_dstore:
            return x, knn_feat
        
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, cfg, num_hidden_layers=5):
        super().__init__()
        self.layer = nn.ModuleList(
            [DecoderLayer(cfg) for _ in range(num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask, clip_his, make_knn_dstore: bool=False):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            output_all_encoded_layers:
        Returns:
        """
        all_decoder_layers = []
        knn_feats = []
        for layer_idx, layer_module in enumerate(self.layer):
            if make_knn_dstore:
                hidden_states, knn_feat =\
                    layer_module(hidden_states, attention_mask, clip_his, make_knn_dstore=make_knn_dstore)
                knn_feats.append(knn_feat)
            
            else:
                hidden_states =\
                    layer_module(hidden_states, attention_mask, clip_his)
            
            all_decoder_layers.append(hidden_states)
        
        if make_knn_dstore:
            return all_decoder_layers, knn_feats
        else:
            return all_decoder_layers


class TrmFeedForward(nn.Module):
    """
    TransformerにおけるFF層
    """

    def __init__(self, cfg):
        super().__init__()
        self.dense_f = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.dense_s = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense_f(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dense_s(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class PredictionHeadTransform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.gelu = nn.GELU()
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, hidden_states):
        """
        (N, L, D)
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class PredictionHead(nn.Module):
    def __init__(self, cfg, bert_model_embedding_weights=None):
        super().__init__()
        self.transform = PredictionHeadTransform(cfg)

        # The output weights are the same as the input embeddings, 
        # but there is　an output-only bias for each token.
        if cfg.share_wd_cls_weight:
            assert bert_model_embedding_weights is not None, (
                "bert_model_embedding_weights should not be None "
                "when setting --share_wd_cls_weight flag to be true"
            )
            assert cfg.hidden_size == bert_model_embedding_weights.size(1), (
                "hidden size has be the same as word embedding size when "
                "sharing word embedding weight and classifier weight"
            )
            self.decoder = nn.Linear(
                bert_model_embedding_weights.size(1),
                bert_model_embedding_weights.size(0),
                bias=False,
            )
            self.decoder.weight = bert_model_embedding_weights
        else:
            self.decoder = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size))

    def forward(self, hidden_states):
        """
        (N, L, D)
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states  # (N, L, vocab_size)