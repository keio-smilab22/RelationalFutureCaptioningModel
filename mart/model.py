
"""
MART model.
"""
import copy
import logging
import math
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard.summary import video

from mart.configs_mart import MartConfig, MartPathConst
from mart.masked_transformer import MTransformer
from mart.loss_caption import LabelSmoothingLoss
from nntrainer.utils_torch import count_parameters


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ACTION_WEIGHT = {596: 12250}

# # default infinity (cfg.inf = 0), works with fp32. this can lead to NaN values in some circumstances
INF = float("inf")


# # this should be "infinite enough" for -INF to give 0 for masked softmax attention values.
# INF = 1e19
# for fp16 need something like 255


def create_mart_model(
    cfg: MartConfig,
    vocab_size: int,
    cache_dir: str = MartPathConst.CACHE_DIR,
    verbose: bool = True,
) -> nn.Module:
    """
    Args:
        cfg: Experiment cfg.
        vocab_size: Vocabulary, calculated in mart as len(train_set.word2idx).
        cache_dir: Cache directory.
        verbose: Print model name and number of parameters.
    Returns:
        MART model.
    """
    cfg.max_position_embeddings = cfg.max_v_len + cfg.max_t_len
    cfg.vocab_size = vocab_size
    if cfg.recurrent:
        logger.info("Use recurrent model - Mine")
        model = RecursiveTransformer(cfg)
    if cfg.use_glove:
        if hasattr(model, "embeddings"):
            logger.info("Load GloVe as word embedding")
            model.embeddings.set_pretrained_embedding(
                torch.from_numpy(
                    torch.load(
                        Path(cache_dir) / f"{cfg.dataset_train.name}_vocab_glove.pt"
                    )
                ).float(),
                freeze=cfg.freeze_glove,
            )
        else:
            logger.warning(
                "This model has no embeddings, cannot load glove vectors into the model"
            )

    # output model properties
    if verbose:
        print(f"Model: {model.__class__.__name__}")
        count_parameters(model)
        if hasattr(model, "embeddings") and hasattr(
            model.embeddings, "word_embeddings"
        ):
            count_parameters(model.embeddings.word_embeddings)

    return model


def gelu(x):
    """
    Implementation of the gelu activation function.
        For information: OpenAI GPT"s gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    Pytorch公式実装のgeluで良さそう
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.
    :Examples:
        >>> model = PositionEncoding(d_model=6, max_len=10, dropout=0)
        >>> test_input1 = torch.zeros(3, 10, 6)
        >>> output1 = model(test_input1)
        >>> output1.size()
        >>> test_input2 = torch.zeros(5, 3, 9, 6)
        >>> output2 = model(test_input2)
        >>> output2.size()
    """

    def __init__(self, n_filters=128, max_len=500):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_filters, 2).float() * -(math.log(10000.0) / n_filters)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[: x.size(-2), :]  # (#x.size(-2), n_filters)
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x


class SelfAttention(nn.Module):
    """
    Attentionの計算
    """

    def __init__(self, cfg):
        super().__init__()
        if cfg.hidden_size % cfg.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (cfg.hidden_size, cfg.num_attention_heads)
            )
        self.num_attention_heads = cfg.num_attention_heads
        self.attention_head_size = int(cfg.hidden_size / cfg.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_w = nn.Linear(cfg.hidden_size, self.all_head_size)
        self.key_w = nn.Linear(cfg.hidden_size, self.all_head_size)
        self.value_w = nn.Linear(cfg.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(cfg.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query, key, value, attention_mask=None):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)
        Returns:
        """
        # only need to mask the dimension where the softmax
        # (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        if attention_mask is not None:
            attention_mask = (
                1 - attention_mask.unsqueeze(1)
            ) * -10000.0  # (N, 1, Lq, L)
        mixed_query_layer = self.query_w(query)
        mixed_key_layer = self.key_w(key)
        mixed_value_layer = self.value_w(value)
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)
        # Take the dot product between "query" and "key"
        # to get the raw attention scores.
        att_w = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        att_w = att_w / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers
            # in BertModel forward() function)
            att_w = att_w + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(att_w)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    """
    TransformerにおけるFF層
    """

    def __init__(self, cfg):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    """
    TransformerにおけるMHA
    """

    def __init__(self, cfg):
        super().__init__()
        self.self = SelfAttention(cfg)
        self.output = SelfOutput(cfg)

    def forward(self, x, attention_mask=None, clip_his=None):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, Lq, L)
        Returns:
        """
        if clip_his is not None:
            self_output = self.self(clip_his, x, x, attention_mask)
        else:
            self_output = self.self(x, x, x, attention_mask)
        att = self.output(self_output, x)
        return att


class Intermediate(nn.Module):
    """
    geluを用いた1層線形変換
    """

    def __init__(self, cfg):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Output(nn.Module):
    """
    GeneratorにおけるFF層
    """

    def __init__(self, cfg):
        super().__init__()
        self.dense = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def make_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=0, decoder=False):
    """
    Args:
        input_mask: (N, L) with `1` indicates valid bits, `0` indicates pad
        max_v_len: int, the first `max_v_len` is for video and its padding, the length
            of the rest of the bits is `max_t_len`. We have L = `max_v_len` + `max_t_len`.
            Note max_v_len may also include the memory len (M), thus max_v_len += M
        max_t_len: int
        memory_len: int, M
    Returns:
    >>> max_v_len_ = 2
    >>> max_t_len_ = 3
    >>> input_mask_ = torch.randn(2, 5)
    >>> make_pad_shifted_mask(input_mask_, max_v_len_, max_t_len_)[0]
    tensor([[1., 1., 0., 0., 0.],
            [1., 1., 0., 0., 0.],
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.]])
    """
    bsz, seq_len = input_mask.shape
    assert max_v_len + max_t_len + memory_len == seq_len
    shifted_mask = input_mask.new_zeros(
        bsz, max_v_len + max_t_len, seq_len
    )  # (N, L, M+L)
    shifted_mask[:, :, : memory_len + max_v_len] = 1
    shifted_mask[:, max_v_len:, memory_len + max_v_len :] = torch.tril(
        input_mask.new_ones(max_t_len, max_t_len), diagonal=0
    )
    if decoder:
        shifted_mask = torch.ones(shifted_mask.size())
    return shifted_mask


def make_pad_shifted_mask(
    input_mask, max_v_len, max_t_len, memory_len=0, decoder=False
):
    """
    input_mask: (N, L),
    """
    shifted_mask = make_shifted_mask(
        input_mask, max_v_len, max_t_len, memory_len=memory_len, decoder=False
    )
    # It's correct to use `input_mask.unsqueeze(1)' instead of
    # `torch.bmm(input_mask.unsqueeze(2), input_mask.unsqueeze(1))'
    # since the rest of the bits are still masked in the subsequent processing steps.
    pad_shifted_mask = shifted_mask * input_mask.unsqueeze(1)
    return pad_shifted_mask


def make_video_only_mask(input_mask, max_v_len):
    video_only_mask = copy.deepcopy(input_mask)
    video_only_mask[:, max_v_len:] = 0
    return video_only_mask


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


class LayerWoMemory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.attention = Attention(cfg)
        self.hidden_intermediate = Intermediate(cfg)
        self.output = Output(cfg)

    def forward(self, hidden_states, attention_mask, clip_feats=None):
        """
        Args:
            prev_m: (N, M, D)
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        Returns:
        """
        max_v_len, max_t_len = self.cfg.max_v_len, self.cfg.max_t_len
        # self-attention, need to shift right
        shifted_self_mask = make_pad_shifted_mask(
            attention_mask, max_v_len, max_t_len
        )  # (N, L, L)
        attention_output = self.attention(hidden_states, shifted_self_mask, clip_feats)
        intermediate_output = self.hidden_intermediate(attention_output)

        return intermediate_output


class EncoderWoMemory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.ModuleList(
            [LayerWoMemory(cfg) for _ in range(cfg.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, clip_feats=None):
        """
        Args:
            prev_ms: [(N, M, D), ] * num_hidden_layers or None at first step.
            Memory states for each layer
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            output_all_encoded_layers:
        Returns:
        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask, clip_feats)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.attention = Attention(cfg)
        self.output = TrmFeedForward(cfg)

    def forward(self, x, attention_mask, clip_his):
        """
        Args:
            prev_m: (N, M, D)
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        Returns:
        """
        max_v_len, max_t_len = self.cfg.max_v_len, self.cfg.max_t_len
        # self-attention, need to shift right
        shifted_self_mask = make_pad_shifted_mask(
            attention_mask, max_v_len, max_t_len, decoder=True
        )  # (N, L, L)
        att = self.attention(x, shifted_self_mask, clip_his)
        layer_output = self.output(att, att)  # (N, L, D)
        return layer_output

class SenseLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.attention = Attention(cfg)
        self.output = TrmFeedForward(cfg)

    def forward(self, x, clip_his):
        """
        Args:
            prev_m: (N, M, D)
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        Returns:
        """
        shifted_self_mask = None
        max_v_len, max_t_len = self.cfg.max_v_len, self.cfg.max_t_len
        # self-attention, need to shift right
        att = self.attention(x, shifted_self_mask, clip_his)
        layer_output = self.output(att, att)  # (N, L, D)
        return layer_output


class Decoder(nn.Module):
    def __init__(self, cfg, num_hidden_layers=5):
        super().__init__()
        self.layer = nn.ModuleList(
            [DecoderLayer(cfg) for _ in range(num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask, clip_his):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            output_all_encoded_layers:
        Returns:
        """
        query_clip = torch.zeros(hidden_states.shape).cuda()
        query_clip = query_clip + clip_his
        all_decoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states =\
                layer_module(hidden_states, attention_mask, clip_his)
            # hidden_states =\
            #     layer_module(hidden_states, query_clip)
            all_decoder_layers.append(hidden_states)
        return all_decoder_layers


class SensorTransformer(nn.Module):
    def __init__(self, cfg, num_hidden_layers=3):
        super().__init__()
        self.layer = nn.ModuleList(
            [SenseLayer(cfg) for _ in range(num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            output_all_encoded_layers:
        Returns:
        """
        pred_sens = []
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states =\
                layer_module(hidden_states, hidden_states)
            # hidden_states =\
            #     layer_module(hidden_states, query_clip)
            pred_sens.append(hidden_states)
        return pred_sens[-1] # (batch_size, 4, 786)


class TrmEncLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.attention = RelationalSelfAttention(cfg)
        # self.attention = Attention(cfg)
        self.output = TrmFeedForward(cfg)

    def forward(self, x):
        """
        Args:
            x: (N, L, D)
        Returns:
        """
        tmp_x = x.clone()
        target = x[:, 1, :].clone()
        target = self.attention(target, tmp_x)
        x[:, 1, :] = target.clone()
        # x = self.attention(x)
        x = self.output(x, x)  # (N, L, D)
        return x


class TimeSeriesEncoder(nn.Module):
    def __init__(self, cfg, num_layers=2):
        super().__init__()
        self.cfg = cfg
        self.pe = PositionEncoding(n_filters=384)
        self.layers = nn.ModuleList([TrmEncLayer(self.cfg) for _ in range(num_layers)])
        self.ff = TrmFeedForward(self.cfg)

    def forward(self, x):
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ff(x, x)
        return x


class EmbeddingsWithVideo(nn.Module):
    """
    Construct the embeddings from word (+ video),
    position and token_type embeddings.
    input_ids (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] filled with [VID]
    video_features (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] as real features, others as zeros
    ==> video features and word embeddings are merged together by summing up.
    """

    def __init__(self, cfg, add_postion_embeddings=True):
        super().__init__()
        """
        add_postion_embeddings: whether to add absolute positional embeddings
        """
        cfg.video_feature_size = 384
        self.add_postion_embeddings = add_postion_embeddings
        self.word_embeddings = nn.Embedding(
            cfg.vocab_size, cfg.word_vec_size, padding_idx=0
        )
        self.word_fc = nn.Sequential(
            nn.LayerNorm(cfg.word_vec_size, eps=cfg.layer_norm_eps),
            nn.Dropout(cfg.hidden_dropout_prob),
            nn.Linear(cfg.word_vec_size, cfg.hidden_size),
            nn.ReLU(True),
            nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps),
        )
        self.video_embeddings = nn.Sequential(
            nn.LayerNorm(cfg.video_feature_size, eps=cfg.layer_norm_eps),
            # nn.Dropout(cfg.hidden_dropout_prob),
            nn.Linear(cfg.video_feature_size, cfg.hidden_size),
            nn.ReLU(True),
            nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps),
        )

        if self.add_postion_embeddings:
            self.position_embeddings = PositionEncoding(
                n_filters=cfg.hidden_size, max_len=cfg.max_position_embeddings * 2
            )
        self.token_type_embeddings = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def set_pretrained_embedding(self, pretrained_embedding, freeze=True):
        """
        Note the from_pretrained does not work in-place, so you need to assign value to the embedding
        """
        assert (
            pretrained_embedding.shape == self.word_embeddings.weight.shape
        )  # ensure equal shape
        self.word_embeddings = nn.Embedding.from_pretrained(
            pretrained_embedding,
            freeze=freeze,
            padding_idx=self.word_embeddings.padding_idx,
        )

    def forward(self, input_ids, video_features, token_type_ids):
        """
        Args:
            input_ids: (N, L)
            video_features: (N, L, D)
            token_type_ids: (N, L, D)
        Returns:
        """
        words_embeddings = self.word_fc(self.word_embeddings(input_ids))
        video_embeddings = self.video_embeddings(video_features)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        words_embeddings += token_type_embeddings
        embeddings = words_embeddings + video_embeddings + token_type_embeddings

        if self.add_postion_embeddings:
            embeddings = self.position_embeddings(embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # (N, L, D)


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


class LMPredictionHead(nn.Module):
    def __init__(self, cfg, bert_model_embedding_weights=None):
        super().__init__()
        self.transform = PredictionHeadTransform(cfg)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
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


class RelationalSelfAttention(nn.Module):
    """
    Relational self-attention (RSA)
    https://arxiv.org/pdf/2111.01673.pdf
    """
    def __init__(self, cfg, m=3):
        super().__init__()
        self.cfg = cfg
        self.m = m
        self.hidden_size = 384
        self.query_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.p = torch.randn((m, self.hidden_size), requires_grad=True).cuda()
        self.h =\
            torch.randn((m * self.hidden_size, m), requires_grad=True).cuda()
        self.g = torch.randn((m, self.hidden_size), requires_grad=True).cuda()
        self.one = torch.ones((m, 1)).cuda()

    def forward(self, target, cont):
        query = self.query_layer(target).reshape(-1, self.hidden_size, 1)
        key = self.key_layer(cont)
        value = self.value_layer(cont)

        # basic kernel
        kernel_v = torch.matmul(self.p, query).reshape(-1, 1, self.m)

        # relational kernel
        q = torch.matmul(self.one, torch.transpose(query, 1, 2))
        x_q = torch.mul(q, key)
        x_q = x_q.reshape((-1, 1, self.m * self.hidden_size))
        kernel_r = torch.matmul(x_q, self.h).reshape(-1, 1, self.m)
        kernel = kernel_v + kernel_r

        # basic context
        # basic_cont = context.clone()

        # relational context
        xg = value.clone()
        xg = torch.transpose(xg, 1, 2)
        _xg = torch.matmul(xg, self.g)
        x_nr = torch.matmul(value, _xg)
        context = x_nr + value

        output = torch.matmul(kernel, context).reshape(-1, self.hidden_size)

        return output


class TimeSeriesMoudule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg.hidden_size = 384
        self.hidden_size = 768
        self.TSEncoder = TimeSeriesEncoder(self.cfg)
        self.expand = nn.Linear(self.cfg.hidden_size, self.hidden_size)
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.cfg.hidden_size = 768
        self.z = torch.randn(1, requires_grad=True).cuda()

    def forward(self, x):
        ts_feats = x.clone().cuda()
        ts_feats = self.TSEncoder(ts_feats)
        ts_feats = self.z * x + (1 - self.z) * ts_feats
        ts_feats = self.expand(ts_feats)
        tmp_feats = ts_feats[:, 1, :].reshape((-1, 1, self.hidden_size))
        tmp_feats = self.layernorm(tmp_feats)
        return ts_feats, tmp_feats


# MART model
class RecursiveTransformer(nn.Module):
    def __init__(self, cfg: MartConfig):
        super().__init__()
        self.cfg = cfg
        self.cfg.vocab_size = 581
        self.z_f = torch.randn(1, requires_grad=True).cuda()
        self.z_p = torch.randn(1, requires_grad=True).cuda()
        self.embeddings = EmbeddingsWithVideo(cfg, add_postion_embeddings=True)
        self.TSModule = TimeSeriesMoudule(cfg)
        self.encoder = EncoderWoMemory(cfg)
        decoder_classifier_weight = (
            self.embeddings.word_embeddings.weight
            if self.cfg.share_wd_cls_weight
            else None
        )
        self.senstrm = SensorTransformer(cfg)
        self.decoder = LMPredictionHead(cfg, decoder_classifier_weight)
        self.transformerdecoder = Decoder(cfg)
        # if self.cfg.label_smoothing != 0:
        #     self.loss_func = LabelSmoothingLoss(
        #         cfg.label_smoothing, cfg.vocab_size, ignore_index=-1
        #     )
        # else:
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        self.contloss_func = nn.CrossEntropyLoss(ignore_index=-1)
        self.actionloss_func = nn.CrossEntropyLoss()
        self.vloss = nn.MSELoss()
        self.valoss = nn.MSELoss()
        self.closs = nn.MSELoss()
        self.cvloss = nn.MSELoss()
        # clipの特徴量の次元
        input_size = 384
        self.size_adjust = nn.Linear(768, 384)
        self.upsampling = nn.Linear(384, 768)
        self.pred_f = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.ReLU(),
            nn.Linear(input_size * 2, input_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size, input_size),
        )
        self.ff = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.Dropout(0.2),
        )

        self.sens_pre_mod = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.LayerNorm(128, eps=cfg.layer_norm_eps),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.LayerNorm(8, eps=cfg.layer_norm_eps),
            nn.Linear(8, 1)
        )
        self.future_loss = nn.MSELoss()
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version
            # which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward_step(
        self, input_ids, video_features, input_masks, token_type_ids, gt_clip=None
    ):
        """
        single step forward in the recursive structure
        """
        video_features = self.size_adjust(video_features)
        self.future_rec = []
        self.future_gt = []
        if gt_clip is None:
            gt_clip = video_features[:, 1:4, :].clone().cuda()
        # preprocess
        clip_feats = torch.zeros(video_features[:, 1:4, :].shape).cuda()
        clip_feats[:, 0:3, :] = video_features[:, 1:4, :].clone()

        future_b = torch.zeros(video_features[:, 3, :].shape)
        future_b = video_features[:, 3, :].clone()
        future_b = self.pred_f(future_b)
        tmp_feat_f = clip_feats[:, 2, :].clone().cuda()
        clip_feats[:, 2, :] = self.z_f * tmp_feat_f + (1 - self.z_f) * future_b

        # past_feats = gt_clip[:, 0, :].reshape((-1, 1, 384)).clone().cuda()
        # tmp_feats = clip_feats[:, 0, :].reshape((-1, 1, 384)).clone().cuda()
        # past_feats = self.z_p * tmp_feats + (1 - self.z_p) * past_feats
        # clip_feats[:, 0, :] = past_feats.reshape((-1, 384))

        # clip_feats = self.ff(clip_feats)

        # Time Series Module
        all_clip_feats, clip_feats = self.TSModule(clip_feats)

        embeddings = self.embeddings(
            input_ids, video_features, token_type_ids
        )  # (N, L, D)
        # clip_his = torch.zeros((embeddings.shape)).cuda()
        # clip_his = clip_his + ts_feats
        encoded_layer_outputs = self.encoder(
            embeddings, input_masks, output_all_encoded_layers=False
        )  # both outputs are list
        decoded_layer_outputs = self.transformerdecoder(
            encoded_layer_outputs[-1], input_masks, clip_feats
        )
        # Sensor Predictor
        all_clip_feats = torch.cat((all_clip_feats, clip_feats), dim=1)
        sense_pre = self.senstrm(all_clip_feats)
        # print(sense_pre.shape)
        # decoded_feat = torch.cat((decoded_layer_outputs[-1], sense_pre), dim=1) # (-1, 29? , 384)
        decoded_feat = decoded_layer_outputs[-1]
        decoded_feat[:,24:28, :] = sense_pre
        # print(sense_pre.shape)
        sense_pre = self.sens_pre_mod(sense_pre)

        prediction_scores = self.decoder(
            decoded_feat
        )  # (N, L, vocab_size)
        future_b = self.upsampling(future_b)
        return encoded_layer_outputs, prediction_scores, future_b, sense_pre.reshape((-1, 4))
        # return encoded_layer_outputs, prediction_scores

    # ver. future
    def forward(
        self,
        input_ids_list,
        video_features_list,
        input_masks_list,
        token_type_ids_list,
        input_labels_list,
        gt_clip=None,
        gt_sens=None
    ):
        """
        Args:
            input_ids_list: [(N, L)] * step_size
            video_features_list: [(N, L, D_v)] * step_size
            input_masks_list: [(N, L)] * step_size with 1 indicates valid bits
            token_type_ids_list: [(N, L)] * step_size, with `0` on the first `max_v_len` bits,
                `1` on the last `max_t_len`
            input_labels_list: [(N, L)] * step_size, with `-1` on ignored positions,
                will not be used when return_memory is True, thus can be None in this case
            return_memory: bool,
        Returns:
        """
        # [(N, M, D)] * num_hidden_layers, initialized internally
        step_size = len(input_ids_list)
        encoded_outputs_list = []  # [(N, L, D)] * step_size
        prediction_scores_list = []  # [(N, L, vocab_size)] * step_size
        future_rec = []
        future_gt = []
        action_score = []
        sense_pre_list = []
        if gt_clip is not None:
            for idx in range(step_size):
                encoded_layer_outputs, prediction_scores, pred_future, sense_pre = self.forward_step(
                    input_ids_list[idx],
                    video_features_list[idx],
                    input_masks_list[idx],
                    token_type_ids_list[idx]
                )
                future_gt.append(gt_clip[idx])
                future_rec.append(pred_future)
                encoded_outputs_list.append(encoded_layer_outputs)
                prediction_scores_list.append(prediction_scores)
                action_score.append(prediction_scores[:, 3, :])
                sense_pre_list.append(sense_pre)
        else:
            for idx in range(step_size):
                encoded_layer_outputs, prediction_scores, sense_pre = self.forward_step(
                    input_ids_list[idx],
                    video_features_list[idx],
                    input_masks_list[idx],
                    token_type_ids_list[idx]
                )
                encoded_outputs_list.append(encoded_layer_outputs)
                prediction_scores_list.append(prediction_scores)
                action_score.append(prediction_scores[:, 3, :])
                sense_pre_list.append(sense_pre)
        # compute loss, get predicted words
        caption_loss = 0.0
        for idx in range(step_size):
            # print(prediction_scores_list[idx].shape)
            snt_loss = self.loss_func(
                prediction_scores_list[idx].view(-1, self.cfg.vocab_size),
                input_labels_list[idx].view(-1),
            )
            gt_action_list = input_labels_list[idx][:, 3]
            act_score_list = action_score[idx].cpu()
            action_loss = 0.0

            # vel, acc, crs, crs_velでMSE
            speed_std = 6.943259466752163
            acc_std = 1.0128755278649304
            crs_std = 105.43048660106768
            # crs_vel_std = 23.557576723588763
            speed_mean = 6.592310560518758
            acc_mean = -0.032466484184198605
            crs_mean = 179.07880361238463
            # crs_vel_mean = 0.09007327456722607
            crs_vel_mean = 0.10629827308128925
            crs_vel_std = 7.364545291989854

            # sensor loss
            sens_loss = 0.0
            sense_pre_list[idx] = sense_pre_list[idx].view(-1, 4, 1)
            gt_sens[idx] = gt_sens[idx].view(-1, 4, 1)
            # print(sense_pre_list[idx].shape)
            # print(gt_sens[idx].shape)
            v_loss = self.vloss(sense_pre_list[idx][:, 0, :], (gt_sens[idx][:, 0, :]- speed_mean) / speed_std)
            va_loss = self.valoss(sense_pre_list[idx][:, 1, :], (gt_sens[idx][:, 1, :]- acc_mean) / acc_std)
            c_loss = self.closs(sense_pre_list[idx][:, 2, :], (gt_sens[idx][:, 2, :] - crs_mean) / crs_std)
            cv_loss = self.cvloss(sense_pre_list[idx][:, 3, :], (gt_sens[idx][:, 3, :] - crs_vel_mean) / crs_vel_std)
            sens_loss += v_loss + va_loss + c_loss + cv_loss
            # print(sens_loss)
            # for actidx in range(len(gt_action_list)):
            #     gt_action = torch.tensor([gt_action_list[actidx]], dtype=int)
            #     gt_idx = gt_action.tolist()
            #     if gt_idx[0] == -1:
            #         continue
            #     if gt_idx[0] in ACTION_WEIGHT:
            #         action_loss += (1 / ACTION_WEIGHT[gt_idx[0]]) * self.actionloss_func(act_score_list[actidx].view(-1, self.cfg.vocab_size), gt_action)
            #     else:
            #         action_loss += (1 / 300) * self.actionloss_func(act_score_list[actidx].view(-1, self.cfg.vocab_size), gt_action)
            cont_loss = 0.0
            tmp_pred_score_list = prediction_scores_list[idx].view(-1, self.cfg.vocab_size)
            tmp_idx_list = input_labels_list[idx].view(-1)
            # for i in range(1, len(tmp_pred_score_list)):
            #     cont_loss += self.contloss_func(tmp_pred_score_list[i].view(-1, self.cfg.vocab_size), tmp_idx_list[i-1].view(-1))
            # for i in range(0, len(tmp_pred_score_list) - 1):
            #     cont_loss += self.contloss_func(tmp_pred_score_list[i].view(-1, self.cfg.vocab_size), tmp_idx_list[i+1].view(-1))
            if gt_clip is not None:
                fut_loss = self.future_loss(future_rec[idx], future_gt[idx])

            # print(snt_loss)
            # print(fut_loss)
            caption_loss += 0.9 * snt_loss + fut_loss + 0.05 * sens_loss
            # print(caption_loss)

        caption_loss /= step_size
        return caption_loss, prediction_scores_list