

# coding : utf-8
"""
UniterImageModelの作成
@author: Shumpei Hatanaka
"""

import json
import copy
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F


class UniterImageEmbeddings(nn.Module):
    def __init__(self, config, img_dim):
        super().__init__()
        self.img_linear = nn.Linear(img_dim, config.uniter_hidden_size)
        self.img_layer_norm = nn.LayerNorm(config.uniter_hidden_size, eps=1e-12)
        self.pos_linear = nn.Linear(config.uniter_pos_dim, config.uniter_hidden_size)
        self.pos_layer_norm = nn.LayerNorm(config.uniter_hidden_size, eps=1e-12)
        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

        # tf naming convention for layer norm
        self.linear = nn.Linear(config.uniter_hidden_size, config.clip_dim)
        self.layer_norm = nn.LayerNorm(config.clip_dim, eps=1e-12)

    def forward(self, img_feats, img_pos_feats, type_embeddings=None, img_masks=None):
        
        if img_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(img_masks.long())
            img_feats = img_feats + mask

        transformed_img = self.img_layer_norm(self.img_linear(img_feats))
        transformed_pos = self.pos_layer_norm(self.pos_linear(img_pos_feats))
        
        if type_embeddings is not None:
            embeddings = transformed_img + transformed_pos + type_embeddings
        
        embeddings = transformed_img + transformed_pos
        embeddings = self.linear(embeddings)
        embeddings = self.layer_norm(embeddings)

        return embeddings

