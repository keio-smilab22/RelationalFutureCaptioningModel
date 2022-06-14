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

import torch
import torch.nn.functional as F
from torch import nn

from mart.beam_search import BeamSearch
from mart.configs_mart import MartConfig
from mart.recursive_caption_dataset import RecursiveCaptionDataset as RCDataset
from nntrainer import utils


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
    input_ids, input_masks, eos_token_id=RCDataset.EOS, pad_token_id=RCDataset.PAD
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
        self, model: nn.Module, cfg: MartConfig, logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.cfg = cfg
        self.logger = logger
        if self.logger is None:
            self.logger = utils.create_logger_without_file(
                "translator", log_level=utils.LogLevelsConst.INFO
            )
    def translate_batch_greedy(
        self,
        input_ids_list,
        video_features_list,
        input_masks_list,
        token_type_ids_list,
        img_feat,
        rt_model,
    ):
        def greedy_decoding_step(
            prev_ms_,
            input_ids,
            video_features,
            input_masks,
            token_type_ids,
            img_feat,
            model,
            max_v_len,
            max_t_len,
            start_idx=RCDataset.BOS,
            unk_idx=RCDataset.UNK,
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
                copied_prev_ms = copy.deepcopy(
                    prev_ms_
                )  # since the func is changing data inside
                _, pred_scores, _ = model.forward_step(
                    input_ids,
                    video_features,
                    input_masks,
                    token_type_ids,
                    img_feat
                )
                # suppress unk token; (N, L, vocab_size)
                pred_scores[:, :, unk_idx] = -1e10
                # next_words = pred_scores.max(2)[1][:, dec_idx]
                next_words = pred_scores[:, dec_idx].max(1)[1]
                next_symbols = next_words

            # compute memory, mimic the way memory is generated at training time
            input_ids, input_masks = mask_tokens_after_eos(input_ids, input_masks)
            return (
                copied_prev_ms,
                input_ids[:, max_v_len:],
            )  # (N, max_t_len == L-max_v_len)

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
                    video_features_list[idx],
                    input_masks_list[idx],
                    token_type_ids_list[idx],
                    img_feat[idx],
                    rt_model,
                    config.max_v_len,
                    config.max_t_len,
                )
                dec_seq_list.append(dec_seq)
            return dec_seq_list

    def translate_batch(
        self,
        model_inputs,
        use_beam=False,
        recurrent=True,
        untied=False,
        xl=False,
        mtrans=False,
    ):
        """
        while we used *_list as the input names, they could be non-list for single sentence decoding case
        """
        if recurrent:
            # input_ids_list, video_features_list, input_masks_list, token_type_ids_list = model_inputs
            # future
            (
                input_ids_list,
                video_features_list,
                input_masks_list,
                token_type_ids_list,
                img_feat,
            ) = model_inputs
            return self.translate_batch_greedy(
                input_ids_list,
                video_features_list,
                input_masks_list,
                token_type_ids_list,
                img_feat,
                self.model,
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
                e1[text_mask] = RCDataset.PAD
                e2[text_mask] = 0  # mark as invalid bits
                video_only_input_ids_list.append(e1)
                video_only_input_masks_list.append(e2)
            return video_only_input_ids_list, video_only_input_masks_list
        else:
            text_mask = segment_ids == 1
            input_ids[text_mask] = RCDataset.PAD
            input_masks[text_mask] = 0
            return input_ids, input_masks

    @classmethod
    def sort_res(cls, res_dict):
        """
        res_dict: the submission json entry `results`
        """
        final_res_dict = {}
        for k, v in list(res_dict.items()):
            # final_res_dict[k] = sorted(v, key=lambda x: float(x["timestamp"][0]))
            # final_res_dict[k] = sorted(v, key=lambda x: float(x["timestamp"]))
            # print(v)
            final_res_dict[k] = sorted(v, key=lambda x: float(x["clip_id"]))
        return final_res_dict
