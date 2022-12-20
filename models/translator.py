"""
Text generation, greedy or beam search.

References:
    Copyright (c) 2017 Jie Lei
    Licensed under The MIT License, see https://choosealicense.com/licenses/mit/
    @inproceedings{lei2020mart,
        title={MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning},
        author={Lei, Jie and Wang, Liwei and Shen, Yelong and Yu, Dong and Berg, Tamara L and Bansal, Mohit},
        booktitle={ACL},
        year={2020}
    }

    History:
    https://github.com/jayleicn/recurrent-transformer
    Current version 2021 https://github.com/gingsi/coot-videotext
"""

import copy
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from datasets.bila import BilaDataset
from utils.configs import Config
from models.beam_search import BeamSearch
from utils import utils
from utils.utils import INFO
import faiss


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def mask_tokens_after_eos(
    input_ids, input_masks, eos_token_id=BilaDataset.EOS, pad_token_id=BilaDataset.PAD
):
    """
    replace values after `[EOS]` with `[PAD]`,
    used to compute memory for next sentence generation
    """
    for row_idx in range(len(input_ids)):
        # possibly more than one `[EOS]`
        # noinspection PyUnresolvedReferences
        cur_eos_idxs = (input_ids[row_idx] == eos_token_id).nonzero(as_tuple=False)
        if len(cur_eos_idxs) != 0:
            cur_eos_idx = cur_eos_idxs[0, 0].item()
            input_ids[row_idx, cur_eos_idx + 1 :] = pad_token_id
            input_masks[row_idx, cur_eos_idx + 1 :] = 0
    return input_ids, input_masks


class Translator(object):
    """
    Load with trained model and handle the beam search.
    """

    def __init__(
        self, model: nn.Module, cfg: Config, logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.cfg = cfg
        self.logger = logger
        if self.logger is None:
            self.logger = utils.create_logger_without_file(
                "translator", log_level=INFO
            )

        # knn
        self.dstore_idx: int = 0
    
    def translate_batch_greedy(
        self,
        input_ids_list,
        img_feats_list,
        txt_feats_list,
        input_masks_list,
        token_type_ids_list,
        bboxes_list,
        bbox_feats_list,
        rt_model,
        make_knn_dstore: bool=False,
        do_knn: bool=False,
    ):
        # ------ knn ------
        if make_knn_dstore:
            d_size = self.cfg.dstore_size
            dstore_keys = np.memmap(self.cfg.dstore_keys_path, dtype=np.float32, mode="r+", shape=(d_size, self.cfg.clip_dim))
            dstore_vals = np.memmap(self.cfg.dstore_vals_path, dtype=np.float32, mode="r+", shape=(d_size, self.cfg.vocab_size))

        def greedy_decoding_step(
            prev_ms_,
            input_ids,
            img_feats,
            txt_feats,
            input_masks,
            token_type_ids,
            bboxes,
            bbox_feats,
            model,
            max_v_len,
            max_t_len,
            start_idx=BilaDataset.BOS,
            unk_idx=BilaDataset.UNK,
            make_knn_dstore: bool=False,
            do_knn: bool=False,
        ):
            """
            RTransformer The first few args are the same to the input to the forward_step func

            Notes:
                1, Copy the prev_ms each word generation step, as the func will modify this value,
                which will cause discrepancy between training and inference
                2, After finish the current sentence generation step, replace the words generated
                after the `[EOS]` token with `[PAD]`. The replaced input_ids should be used to generate
                next memory state tensor.
            """
            bsz = len(input_ids)
            next_symbols = torch.LongTensor([start_idx] * bsz)  # (N, )
            
            for dec_idx in range(max_v_len, max_v_len + max_t_len):
                input_ids[:, dec_idx] = next_symbols
                input_masks[:, dec_idx] = 1
                # since the func is changing data inside
                copied_prev_ms = copy.deepcopy(prev_ms_)
                
                if make_knn_dstore or do_knn:
                    _, pred_scores, _, knn_feats = model.forward_step(
                        input_ids,
                        img_feats,
                        txt_feats,
                        input_masks,
                        token_type_ids,
                        bboxes,
                        bbox_feats,
                        make_knn_dstore=(make_knn_dstore or do_knn),
                    )
                else:
                    _, pred_scores, _ = model.forward_step(
                    input_ids,
                    img_feats,
                    txt_feats,
                    input_masks,
                    token_type_ids,
                    bboxes,
                    bbox_feats,
                    make_knn_dstore=make_knn_dstore
                )
                
                # suppress unk token; (N, L, vocab_size)
                pred_scores[:, :, unk_idx] = -1e10

                if make_knn_dstore:
                    knn_keys = knn_feats[:, dec_idx, :] # (16, 768) <- (16, 63, 768)
                    knn_vals = pred_scores[:, dec_idx, :].cpu().detach().numpy().copy() # (16, 291)

                    shape = knn_keys.shape
                    dstore_keys[self.dstore_idx:self.dstore_idx+shape[0]] = knn_keys
                    dstore_vals[self.dstore_idx:self.dstore_idx+shape[0]] = knn_vals

                    self.dstore_idx += shape[0]

                if do_knn:
                    d_size = self.cfg.dstore_size
                    keys, vals = self.get_knn_feats(self.cfg.dstore_keys_path, self.cfg.dstore_vals_path, d_size, self.cfg.dstore_id_num)

                    DataBase = faiss.IndexFlatL2(self.cfg.clip_dim)
                    DataBase.add(keys)

                    hidden_feats = knn_feats[:, dec_idx, :]

                    D, I = DataBase.search(hidden_feats, self.cfg.k_num) # (16, num_k)
                    batch_size = knn_feats.shape[0]
                    
                    knn_preds = np.stack([vals[I[i]] for i in range(batch_size)]) # (B,num_k,291)
                    
                    knn_origin = False
                    alpha = self.cfg.alpha
                    if knn_origin:
                        knn_preds_agg = np.sum(knn_preds, axis=1)/knn_preds.shape[0]
                        knn_preds_agg = torch.from_numpy(knn_preds_agg).to("cuda")
                        pred_scores = (1-alpha)*pred_scores[:, dec_idx] + alpha*knn_preds_agg
                    else:
                        knn_preds = torch.from_numpy(knn_preds)
                        knn_preds = self.make_knn_preds(knn_preds, torch.from_numpy(D))
                        pred_scores = (1-alpha)*pred_scores[:,dec_idx] + alpha*knn_preds.to('cuda')
                
                if do_knn:
                    next_words = pred_scores.max(1)[1]
                else:
                    next_words = pred_scores[:, dec_idx].max(1)[1] # (B,)
                next_symbols = next_words

            # compute memory, mimic the way memory is generated at training time
            input_ids, input_masks = mask_tokens_after_eos(input_ids, input_masks)
            return (copied_prev_ms, input_ids[:, max_v_len:],)

        input_ids_list, input_masks_list = self.prepare_video_only_inputs(
            input_ids_list, input_masks_list, token_type_ids_list
        )
        for cur_input_masks in input_ids_list:
            assert (
                torch.sum(cur_input_masks[:, self.cfg.max_v_len + 1 :]) == 0
            ), "Initially, all text tokens should be masked"

        config = rt_model.cfg
        with torch.no_grad():
            prev_ms = [None] * config.num_hidden_layers
            step_size = len(input_ids_list)
            dec_seq_list = []
            for idx in range(step_size):
                prev_ms, dec_seq = greedy_decoding_step(
                    prev_ms,
                    input_ids_list[idx],
                    img_feats_list[idx],
                    txt_feats_list[idx],
                    input_masks_list[idx],
                    token_type_ids_list[idx],
                    bboxes_list[idx],
                    bbox_feats_list[idx],
                    rt_model,
                    config.max_v_len,
                    config.max_t_len,
                    make_knn_dstore=make_knn_dstore,
                    do_knn=do_knn,
                )
                dec_seq_list.append(dec_seq)
            return dec_seq_list
    
    def make_knn_preds(
        self,
        knn_preds: torch.Tensor,
        distance: np.ndarray, # (6, 10)
    ):
        B, K, vocab_size = knn_preds.shape

        # (1) knnでとってきた確率に対してargmaxをとる
        pred_idx = torch.argmax(knn_preds, dim=2) # (B,k,291) -> (B,k)

        # (2) 距離を温度Tで割る
        distance = torch.exp(-(distance / self.cfg.knn_temperature))

        # (3) knnの予測値を計算する 
        preds = torch.zeros((B, vocab_size))
        for b in range(B):
            for k in range(K):
                preds[b][pred_idx[b][k]] = distance[b][k]
        
        return preds
    
    def get_knn_feats(
        self,
        kpath: str,
        vpath: str,
        dsize: int,
        Numid: int,
    ):
        keys_memmap = np.memmap(kpath, dtype=np.float32, mode="r+", shape=(dsize, self.cfg.clip_dim))
        key_numpy = np.zeros((self.cfg.dstore_size, self.cfg.clip_dim), dtype=np.float32)
        key_numpy = keys_memmap[:]
        keys = np.zeros((Numid, self.cfg.clip_dim), dtype=np.float32)
        keys[:] = key_numpy[:Numid]
        keys = keys.astype(np.float32)

        vals_memmap = np.memmap(vpath, dtype=np.float32, mode="r+", shape=(dsize, self.cfg.vocab_size))
        vals_numpy = np.zeros((self.cfg.dstore_size, self.cfg.vocab_size), dtype=np.float32)
        vals_numpy = vals_memmap[:]
        vals = np.zeros((Numid, self.cfg.vocab_size), dtype=np.float32)
        vals[:] = vals_numpy[:Numid]
        vals = vals.astype(np.float32)

        return torch.from_numpy(keys), torch.from_numpy(vals)


    def translate_batch(
        self,
        model_inputs,
        use_beam: bool=False,
        make_knn_dstore: bool=False,
        do_knn: bool=False,
    ):
        """
        while we used *_list as the input names, they could be non-list for single sentence decoding case
        """
        (
            input_ids_list,
            img_feats_list,
            txt_feats_list,
            input_masks_list,
            token_type_ids_list,
            bboxes_list,
            bbox_feats_list,
        ) = model_inputs
        
        return self.translate_batch_greedy(
            input_ids_list,
            img_feats_list,
            txt_feats_list,
            input_masks_list,
            token_type_ids_list,
            bboxes_list,
            bbox_feats_list,
            self.model,
            make_knn_dstore=make_knn_dstore,
            do_knn=do_knn,
        )

    @classmethod
    def prepare_video_only_inputs(cls, input_ids, input_masks, segment_ids):
        """
        replace text_ids (except `[BOS]`) in input_ids with `[PAD]` token, for decoding.
        This function is essential!!!
        Args:
            input_ids: (N, L) or [(N, L)] * step_size
            input_masks: (N, L) or [(N, L)] * step_size
            segment_ids: (N, L) or [(N, L)] * step_size
        """
        if isinstance(input_ids, list):
            video_only_input_ids_list = []
            video_only_input_masks_list = []
            for e1, e2, e3 in zip(input_ids, input_masks, segment_ids):
                text_mask = e3 == 1  # text positions (`1`) are replaced
                e1[text_mask] = BilaDataset.PAD
                e2[text_mask] = 0  # mark as invalid bits
                video_only_input_ids_list.append(e1)
                video_only_input_masks_list.append(e2)
            return video_only_input_ids_list, video_only_input_masks_list
        else:
            text_mask = segment_ids == 1
            input_ids[text_mask] = BilaDataset.PAD
            input_masks[text_mask] = 0
            return input_ids, input_masks

    @classmethod
    def sort_res(cls, res_dict):
        """
        res_dict: the submission json entry `results`
        """
        final_res_dict = {}
        for k, v in list(res_dict.items()):
            final_res_dict[k] = sorted(v, key=lambda x: float(x["clip_id"]))
        return final_res_dict
