import logging

import torch
from torch import nn

from losses.loss import LabelSmoothingLoss
from utils.utils import count_parameters
from losses.loss import CLIPloss
from utils.configs import Config
from models.embedder import ResEmbedder, MultiModalEmbedding, ConvNeXtEmbedder, BaseEmbedder, CLIPEmbedder
from models.encoder import RSAEncoder, TransformerEncoder, CrossAttentionEncoder
from models.decoder import TransformerDecoder, PredictionHead
from models.uniter import UniterImageEmbeddings


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ACTION_WEIGHT = {111: 131, 94: 628} 

# # default infinity (cfg.inf = 0), works with fp32. this can lead to NaN values in some circumstances
INF = float("inf")


class RecursiveTransformer(nn.Module):
    def __init__(
        self,
        cfg: Config,
        vocab_size: int
    ):
        super().__init__()
        self.cfg = cfg
        self.cfg.vocab_size = vocab_size
        self.idx = 0

        # feats for object detection
        self.uniter_embedding = UniterImageEmbeddings(cfg, cfg.uniter_img_dim)

        # feats for img and txt
        self.txt_embedder = nn.Linear(cfg.clip_dim, cfg.clip_dim)
        self.embeddings = MultiModalEmbedding(cfg, add_postion_embeddings=True)
        self.CrossAttention = CrossAttention(cfg)
        self.RSAEncoder = RSAEncoder(cfg)
        self.TextEncoder = TransformerEncoder(cfg)
        
        if self.cfg.share_wd_cls_weight:
            decoder_classifier_weight = self.embeddings.word_embeddings.weight
        else:
            decoder_classifier_weight = None
        
        self.transformerdecoder = TransformerDecoder(cfg)
        self.decoder = PredictionHead(cfg, decoder_classifier_weight)
        
        # loss
        if self.cfg.label_smoothing != 0:
            self.loss_func = \
                LabelSmoothingLoss(cfg.label_smoothing, cfg.vocab_size, ignore_index=-1)
        else:
            self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        self.actionloss_func = nn.CrossEntropyLoss()
        self.rec_loss = nn.MSELoss()
        self.cliploss = CLIPloss(hidden_dim=cfg.hidden_size, max_t_length=cfg.max_t_len)
        
        # init weigh
        self.apply(self.init_bert_weights)


    def init_bert_weights(self, module):
        """
        Summary:
            init weights
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
        self,
        input_ids: torch.Tensor,
        img_feats: torch.Tensor,
        txt_feats: torch.Tensor,
        input_masks: torch.Tensor,
        token_type_ids: torch.Tensor,
        bboxes: torch.Tensor,
        bbox_feats: torch.Tensor,
    ):
        """
            singleure step forward in the recursive struct
        Args:
            input_ids (torch.Tensor):
            img_feats (torch.Tensor):
            txt_feats (torch.Tensor):
            input_masks (torch.Tensor):
            token_type_ids (torch.Tensor):
        """
        # feats for object detection
        detection_feats = self.uniter_embedding(bbox_feats, bboxes) # (B, XX, D) XX=可変

        # 特徴抽出
        txt_feats = self.txt_embedder(txt_feats)
        # cross Attention
        img_feats = self.CrossAttention(img_feats, detection_feats) # (B, Lv, D)
        
        # CLS and BOS token
        B,_,D = img_feats.shape
        cls_token = torch.zeros((B,1,D), requires_grad=True).cuda()
        bos_token = torch.zeros((B,1,D), requires_grad=True).cuda()
        # img feats cat
        img_feats = torch.cat((cls_token, img_feats, bos_token), dim=1)

        # cat: (B,Lv,D) + (B,Lt,D) -> (B,Lv+Lt,D)
        features = torch.cat((img_feats[:,:self.cfg.max_v_len, :], txt_feats), dim=1)
        
        # 再構成用
        rec_feat = features[:,1,:].clone()
        # 画像only (B, Lv-2, D)
        img_feats = features[:,1:self.cfg.max_v_len-1,:].clone()

        # Time Series Module
        img_feats, _  = self.RSAEncoder(img_feats)

        embeddings = self.embeddings(input_ids, features, token_type_ids)
        
        encoded_layer_outputs = self.TextEncoder(
            embeddings, input_masks, output_all_encoded_layers=False
        )
        # encoded_layer_ouputs[-1]: (16,63,768)
        # img_feats: (16,1,768)
        decoded_layer_outputs = self.transformerdecoder(
            encoded_layer_outputs[-1], input_masks, img_feats
        )
        prediction_scores = self.decoder(
            decoded_layer_outputs[-1]
        )  # (N, L, vocab_size)
        return encoded_layer_outputs, prediction_scores, rec_feat

    def forward(
        self,
        input_ids_list,
        img_feats_list,
        txt_feats_list,
        input_masks_list,
        token_type_ids_list,
        input_labels_list,
        gt_rec=None,
        bbox_list=None,
        bbox_feats_list=None,
    ):
        """
        Args:
            input_ids_list: [(N, L)] * step_size
            input_masks_list: [(N, L)] * step_size with 1 indicates valid bits
            token_type_ids_list: [(N, L)] * step_size, with `0` on the first `max_v_len` bits,
                `1` on the last `max_t_len`
            input_labels_list: [(N, L)] * step_size, with `-1` on ignored positions,
                will not be used when return_memory is True, thus can be None in this case
            return_memory: bool,

        Returns:
        """
        step_size = len(input_ids_list) # 1
        encoded_outputs_list = []  # [(N, L, D)] * step_size
        prediction_scores_list = []  # [(N, L, vocab_size)] * step_size
        pred_reconst = []
        gt_reconst = []
        action_score = []

        if gt_rec is not None:
            for idx in range(step_size):
                encoded_layer_outputs, prediction_scores, pred_tmp = self.forward_step(
                    input_ids_list[idx],
                    img_feats_list[idx],
                    txt_feats_list[idx],
                    input_masks_list[idx],
                    token_type_ids_list[idx],
                    bbox_list[idx],
                    bbox_feats_list[idx],
                )
                gt_reconst.append(gt_rec[idx])
                pred_reconst.append(pred_tmp)
                encoded_outputs_list.append(encoded_layer_outputs)
                prediction_scores_list.append(prediction_scores)
                action_score.append(prediction_scores[:, 7, :])
        else:
            for idx in range(step_size):
                encoded_layer_outputs, prediction_scores = self.forward_step(
                    input_ids_list[idx],
                    img_feats_list[idx],
                    txt_feats_list[idx],
                    input_masks_list[idx],
                    token_type_ids_list[idx],
                    bbox_list[idx],
                    bbox_feats_list[idx],
                )
                encoded_outputs_list.append(encoded_layer_outputs)
                prediction_scores_list.append(prediction_scores)
                action_score.append(prediction_scores[:, 7, :])
        
        # compute loss, get predicted words
        caption_loss = 0.0
        for idx in range(step_size):
            loss_CE = self.loss_func(
                prediction_scores_list[idx].view(-1, self.cfg.vocab_size),
                input_labels_list[idx].view(-1),
            )
            clip_loss = 0.0
            clip_loss += self.cliploss(pred_reconst[idx], encoded_outputs_list[idx][0][:, self.cfg.max_v_len:, :])
            if gt_rec is not None:
                rec_loss = self.rec_loss(pred_reconst[idx].reshape(-1, 16, 16, 3), gt_reconst[idx] / 255.)

            # caption_loss += 15 * snt_loss + 500 * rec_loss + 4 * clip_loss
            caption_loss += 15 * loss_CE + 4 * clip_loss
        caption_loss /= step_size
        return caption_loss, prediction_scores_list, loss_CE, rec_loss, clip_loss

class CrossAttention(nn.Module):
    """
    Summary:
        calc attention between camera rgbd and taregt rgbd
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.ca_embedder == "Base":
            self.img_embedder = BaseEmbedder(cfg)
        elif cfg.ca_embedder == "CLIP":
            self.img_embedder = CLIPEmbedder(cfg)
        elif cfg.ca_embedder == "Res":
            self.img_embedder = ResEmbedder(cfg)
        elif cfg.ca_embedder == "ConvNeXt":
            self.img_embedder = ConvNeXtEmbedder()

        self.img_encoder = CrossAttentionEncoder(cfg)
        self.detect_encoder = CrossAttentionEncoder(cfg)

    def forward(self, img_feats: torch.Tensor, detect_feats: torch.Tensor):
        """
        Args:
            img_feats(torch.Tensor): (B, 6(4), H, W, C)
            if max_v_len==8 4 else 2
        """
        img_feats = self.img_embedder(img_feats)
        
        img_feats = self.img_encoder(hidden_states=img_feats, source_kv=detect_feats) #(B,4(2),D)
        detect_feats = self.detect_encoder(hidden_states=detect_feats, source_kv=img_feats) #(B,4(2),D)

        # concat
        img_feats = torch.cat((img_feats, detect_feats), dim=1) #(B, 6(4), D)

        return img_feats

def create_model(
    cfg: Config,
    vocab_size: int,
    verbose: bool = True,
) -> nn.Module:
    """
    Args:
        cfg: config
        vocab_size: len(train_set.word2idx).
        verbose: Print model name and number of parameters.
    """
    cfg.max_position_embeddings = cfg.max_v_len + cfg.max_t_len
    cfg.vocab_size = vocab_size

    model = RecursiveTransformer(cfg, vocab_size=vocab_size)

    # output model properties
    if verbose: # true
        print(f"Model: {model.__class__.__name__}")
        count_parameters(model)
        if hasattr(model, "embeddings") and hasattr(model.embeddings, "word_embeddings"):
            count_parameters(model.embeddings.word_embeddings)

    return model