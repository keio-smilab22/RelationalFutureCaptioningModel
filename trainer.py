"""
Trainer for retrieval training and validation. Holds the main training loop.
"""
import os
import datetime
import json
import logging
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils import data
from tqdm import tqdm
import wandb

from datasets.bila import BilaDataset, prepare_batch_inputs
from utils.utils import get_reference_files
from utils.configs import Config, MetersConst as MMeters
from utils.manager import FilesHandler, ModelManager
from metrics.evaluate_language import evaluate_language_files
from metrics.evaluate_repetition import evaluate_repetition_files
from metrics.evaluate_stats import evaluate_stats_files
from metrics.metric import TRANSLATION_METRICS, TextMetricsConst, TextMetricsConstEvalCap
from optim.optim import BertAdam, EMA
from models.translator import Translator

from utils.utils import INFO
from utils.configs import TrainerState
from utils import utils
from utils.utils import MetricComparisonConst, dump_yaml_config_file
from optim import lr_scheduler
from metrics import metric
from metrics.metric import DefaultMetricsConst as Metrics


def cal_performance(pred, gt):
    pred = pred.max(2)[1].contiguous().view(-1)
    gt = gt.contiguous().view(-1)
    valid_label_mask = gt.ne(BilaDataset.IGNORE)
    pred_correct_mask = pred.eq(gt)
    n_correct = pred_correct_mask.masked_select(valid_label_mask).sum().item()
    return n_correct


# only log the important ones to console
TRANSLATION_METRICS_LOG = ["Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "re4"]


class Trainer:
    """
    Trainer

    Args:
        cfg: Loaded configuration instance.
        model: Model.
        run_name: Experiment run.
        train_loader_length: Length of the train loader, required for some LR schedulers.
        log_dir: Directory to put results.
        log_level: Log level. None will default to INFO = 20 if a new logger is created.
        logger: Logger. With the default None, it will be created by the trainer.
        reset: Delete entire experiment and restart from scratch.
        load_best: Whether to load the best epoch (default loads last epoch to continue training).
        load_epoch: Whether to load a specific epoch.
        load_model: Load model given by file path.
        is_test: Removes some parts that are not needed during inference for speedup.
        data_dir: Folder with ground truth captions.
    """

    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        run_name: str,
        train_loader_length: int,
        log_dir: str = "results",
        log_level: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        reset: bool = False,
        load_best: bool = False,
        load_epoch: Optional[int] = None,
        load_model: Optional[str] = None,
        is_test: bool = False,
        data_dir: str = "data",
        show_log: bool = False,
    ):
        assert "_" not in run_name, f"Run name {run_name} must not contain underscores."
        
        self.model = model

        # knn
        self.dstore_idx = 0

        # ファイル関連の設定
        exp = FilesHandler(
            run_name=run_name,
            log_dir=log_dir,
            data_dir=data_dir
        )
        # フォルダの作成
        exp.setup_dirs(reset=reset)

        # settings
        self.is_test: bool = is_test
        # save model manager
        self.model_mgr: ModelManager = ModelManager(cfg, model)
        # create empty trainer state
        self.state = TrainerState()
        # save config
        self.cfg: Config = cfg
        # create experiment helper for directories
        self.exp: FilesHandler = exp


        # loggingの設定
        assert logger is None or log_level is None, "Cannot specify both loglevel and logger together."
        if logger is None:
            if log_level is None:
                self.log_level = INFO
            else:
                self.log_level = log_level
            self.logger = utils.create_logger(utils.LOGGER_NAME, log_dir=self.exp.path_logs, log_level=self.log_level)
        else:
            self.logger = logger
            self.log_level = self.logger.level
        
        # setup devices
        cudnn.enabled = self.cfg.cudnn_enabled
        if not self.cfg.use_cuda:
            self.cfg.fp16_train = False
        
        # setup grad scaler if needed for fp16
        self.grad_scaler: Optional[GradScaler] = None
        if self.cfg.fp16_train:
            self.grad_scaler: Optional[GradScaler] = GradScaler()
        
        # logs some infos
        if show_log:
            self.logger.info(f"Running on cuda: {self.cfg.use_cuda} "
                                f"gpus found: {torch.cuda.device_count()}, fp16 amp: {self.cfg.fp16_train}.")
        
        # move models to cuda
        for model in self.model_mgr.model_dict.values():
            try:
                if self.cfg.use_cuda:
                    if not torch.cuda.is_available():
                        raise RuntimeError(
                            "CUDA requested but not available! Use --no_cuda to run on CPU.")
                    model = model.cuda()
            except RuntimeError as e:
                raise RuntimeError(f"RuntimeError when putting model {type(model)} to cuda with DataParallel "
                                    f"{model.__class__.__name__}") from e
        
        # create metrics writer
        self.metrics = metric.MetricsWriter(self.exp)

        # seedの表示
        self.logger.info(f"Random seed: {self.cfg.random_seed}")

        # 使用したconfig fileを保存
        dump_yaml_config_file(self.exp.path_base / 'config.yaml', self.cfg.config_orig)
        
        # setup automatic checkpoint loading. this will be parsed in self.hook_post_init()
        ep_nums = self.exp.get_existing_checkpoints()
        self.load = False
        self.load_ep = -1
        self.load_model = load_model
        
        if self.load_model:
            assert not load_epoch, (
                "When given filepath with load_model, --load_epoch must not be set.")
            self.load = True
        
        # automatically find best epoch otherwise
        elif len(ep_nums) > 0:
            if load_epoch:
                # load given epoch
                assert not load_best, "Load_epoch and load_best cannot be set at the same time."
                self.load_ep = load_epoch
            elif load_best:
                # load best epoch
                self.logger.info("Load best checkpoint...")
                best_ep = self.exp.find_best_epoch()
                if best_ep == -1:
                    # no validation done yet, load last
                    self.load_ep = ep_nums[-1]
                else:
                    self.load_ep = best_ep
                self.logger.info(f"Best ckpt to load: {self.load_ep}")
                self.load = True
            else:
                # load last epoch
                self.load_ep = ep_nums[-1]
                self.logger.info(f"Last ckpt to load: {self.load_ep}")
                self.load = True
        else:
            self.logger.info("No checkpoints found, starting from scratch.")
        
        # Per-epoch metrics where the average is not important.
        self.metrics.add_meter(Metrics.TRAIN_EPOCH, use_avg=False)
        self.metrics.add_meter(Metrics.TIME_TOTAL, use_avg=False)
        self.metrics.add_meter(Metrics.TIME_VAL, use_avg=False)
        self.metrics.add_meter(Metrics.VAL_LOSS, use_avg=False)
        self.metrics.add_meter(Metrics.VAL_BEST_FIELD, use_avg=False)

        # Per-step metrics
        self.metrics.add_meter(Metrics.TRAIN_LR, per_step=True, use_avg=False)
        self.metrics.add_meter(Metrics.TRAIN_GRAD_CLIP, per_step=True, reset_avg_each_epoch=True)
        self.metrics.add_meter(Metrics.TRAIN_LOSS, per_step=True, reset_avg_each_epoch=True)

        # Per-step Memory-RAM Profiling
        self.metrics.add_meter(Metrics.PROFILE_GPU_MEM_USED, per_step=True)
        self.metrics.add_meter(Metrics.PROFILE_GPU_LOAD, per_step=True)
        self.metrics.add_meter(Metrics.PROFILE_RAM_USED, per_step=True)
        self.metrics.add_meter(Metrics.PROFILE_GPU_MEM_TOTAL, per_step=True, use_avg=False)
        self.metrics.add_meter(Metrics.PROFILE_RAM_TOTAL, per_step=True, use_avg=False)

        # Step-based metrics for time, we only care about the total average
        self.metrics.add_meter(Metrics.TIME_STEP_FORWARD, per_step=True, use_value=False)
        self.metrics.add_meter(Metrics.TIME_STEP_BACKWARD, per_step=True, use_value=False)
        self.metrics.add_meter(Metrics.TIME_STEP_TOTAL, per_step=True, use_value=False)
        self.metrics.add_meter(Metrics.TIME_STEP_OTHER, per_step=True, use_value=False)

        # compute steps per epoch
        self.train_loader_length = train_loader_length

        # The following fields must be set by the inheriting trainer. In special cases (like multiple optimizers
        # with GANs), override methods get_opt_state and set_opt_state instead.
        self.optimizer: Optimizer = None
        self.lr_scheduler: lr_scheduler.LRScheduler = None

        # setup timers and other stuff that does not need to be saved (temporary trainer state)
        self.timer_step: float = 0
        self.timer_step_forward: float = 0
        self.timer_step_backward: float = 0
        self.timer_train_start: float = 0
        self.timer_train_epoch: float = 0
        self.timer_val_epoch: float = 0
        self.timedelta_step_forward: float = 0
        self.timedelta_step_backward: float = 0
        self.steps_per_epoch: int = 0
        
        # ---------- additional metrics ----------
        # train loss and accuracy
        self.metrics.add_meter(MMeters.TRAIN_LOSS_PER_WORD, use_avg=False)
        self.metrics.add_meter(MMeters.TRAIN_ACC, use_avg=False)
        self.metrics.add_meter(MMeters.VAL_LOSS_PER_WORD, use_avg=False)
        self.metrics.add_meter(MMeters.VAL_ACC, use_avg=False)

        # track gradient clipping manually
        self.metrics.add_meter(MMeters.GRAD, per_step=True, reset_avg_each_epoch=True)

        # translation metrics (bleu etc.)
        for meter_name in TRANSLATION_METRICS.values():
            self.metrics.add_meter(meter_name, use_avg=False)


        self.optimizer = None
        self.lr_scheduler = None
        self.ema = EMA(cfg.ema_decay)
        self.best_epoch = 0
        # skip optimizer if not training
        if not self.is_test:
            # Prepare optimizer
            param_optimizer = list(model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.01,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            if cfg.ema_decay > 0:
                # register EMA params
                self.logger.info(
                    f"Registering {sum(p.numel() for p in model.parameters())} params for EMA"
                )
                all_names = []
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        self.ema.register(name, p.data)
                    all_names.append(name)
                self.logger.debug("\n".join(all_names))

            num_train_optimization_steps = train_loader_length * cfg.train.num_epochs
            self.optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=cfg.lr,
                warmup=cfg.lr_warmup_proportion,
                t_total=num_train_optimization_steps,
                e=cfg.eps,
                schedule="warmup_linear",
            )

        self.translator = Translator(self.model, self.cfg, logger=self.logger)

        # post init hook for checkpoint loading
        self.hook_post_init()

        if self.load and not self.load_model:
            # reload EMA weights from checkpoint (the shadow) and save the model parameters (the original)
            ema_file = self.exp.get_models_file_ema(self.load_ep)
            self.logger.info(f"Update EMA from {ema_file}")
            self.ema.set_state_dict(torch.load(str(ema_file)))
            self.ema.assign(self.model, update_model=False)

        # disable ema when loading model directly or when decay is 0 / -1
        if self.load_model or cfg.ema_decay <= 0:
            self.ema = None

        self.train_steps = 0
        self.val_steps = 0
        self.test_steps = 0
        self.beforeloss = 0.0

    def train_model(
        self,
        train_loader: data.DataLoader,
        val_loader: data.DataLoader,
        test_loader: data.DataLoader,
        datatype: str = "bila",
        use_wandb: bool = False,
        show_log: bool = False,
        make_knn_dstore: bool = False,
        do_knn: bool = False,
    ) -> None:
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb_name = f"{datatype}_{self.cfg.max_t_len}_{self.cfg.max_v_len}_make_knn"
            wandb.init(name=wandb_name, project="BilaS")
        
        # set start epoch and time & show log
        self.hook_pre_train(show_log)
        
        for _epoch in tqdm(range(self.state.current_epoch, self.cfg.train.num_epochs)):
            # set models to train, time book-keeping
            self.hook_pre_train_epoch(show_log)

            # check exponential moving average
            if (
                self.ema is not None
                and self.state.current_epoch != 0
                and self.cfg.ema_decay != -1
            ):
                # use normal parameters for training, not EMA model
                self.ema.resume(self.model)
            
            torch.autograd.set_detect_anomaly(True)

            total_loss = 0
            n_word_total = 0
            n_word_correct = 0
            num_steps = 0
            batch_loss = 0.0
            batch_snt_loss = 0.0
            batch_rec_loss = 0.0
            batch_clip_loss = 0.0

            for step, batch in enumerate(tqdm(train_loader)):
                # hook for step timing
                self.hook_pre_step_timer()

                self.optimizer.zero_grad()
                with autocast(enabled=self.cfg.fp16_train):
                    # input to cuda
                    batched_data = [
                        prepare_batch_inputs(
                            step_data,
                            use_cuda=self.cfg.use_cuda,
                            non_blocking=self.cfg.cuda_non_blocking,
                        )
                        for step_data in batch[0]
                    ]

                    # dict -> list
                    input_ids_list = [e["input_ids"] for e in batched_data]
                    img_feats_list = [e['img_feats'] for e in batched_data]
                    txt_feats_list = [e['txt_feats'] for e in batched_data]
                    input_masks_list = [e["input_mask"] for e in batched_data]
                    token_type_ids_list = [e["token_type_ids"] for e in batched_data]
                    input_labels_list = [e["input_labels"] for e in batched_data]
                    gt_rec = [e["gt_rec"] for e in batched_data]
                    bboxes_list = [e["bboxes"] for e in batched_data]
                    bbox_feats_list = [e["bbox_feats"] for e in batched_data]

                    if self.cfg.debug:
                        cur_data = batched_data[step]
                        self.logger.info(
                            "input_ids \n{}".format(cur_data["input_ids"][step]))
                        self.logger.info(
                            "input_mask \n{}".format(cur_data["input_mask"][step]))
                        self.logger.info(
                            "input_labels \n{}".format(
                                cur_data["input_labels"][step]))
                        self.logger.info(
                            "token_type_ids \n{}".format(
                                cur_data["token_type_ids"][step]))


                    loss, pred_scores_list, snt_loss, rec_loss, clip_loss = self.model(
                        input_ids_list,
                        img_feats_list,
                        txt_feats_list,
                        input_masks_list,
                        token_type_ids_list,
                        input_labels_list,
                        gt_rec,
                        bboxes_list,
                        bbox_feats_list,
                    )

                    self.train_steps += 1
                    num_steps += 1
                    batch_loss += loss
                    batch_snt_loss += snt_loss
                    batch_rec_loss += rec_loss
                    batch_clip_loss += clip_loss
                
                # hook for step timing
                self.hook_post_forward_step_timer()

                grad_norm = None
                if self.cfg.fp16_train:
                    # with fp16 amp
                    self.grad_scaler.scale(loss).backward()
                    if self.cfg.train.clip_gradient != -1:
                        # gradient clipping
                        self.grad_scaler.unscale_(self.optimizer)
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.train.clip_gradient
                        )
                    # gradient scaler realizes if gradients have been unscaled already and doesn't do it again.
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    # with regular float32
                    loss.backward()
                    if self.cfg.train.clip_gradient != -1:
                        # gradient clipping
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.train.clip_gradient
                        )
                    self.optimizer.step()
                
                # update model parameters with ema
                if self.ema is not None:
                    self.ema(self.model, self.state.total_step)

                # keep track of loss, accuracy, gradient norm
                total_loss += loss.item()
                n_correct = 0
                n_word = 0
                
                for pred, gt in zip(pred_scores_list, input_labels_list):
                    n_correct += cal_performance(pred, gt)
                    valid_label_mask = gt.ne(BilaDataset.IGNORE)
                    n_word += valid_label_mask.sum().item()
                n_word_total += n_word
                n_word_correct += n_correct
                if grad_norm is not None:
                    self.metrics.update_meter(MMeters.GRAD, grad_norm)

                if self.cfg.debug:
                    break

                additional_log = f" Grad {self.metrics.meters[MMeters.GRAD].avg:.2f}"
                self.hook_post_backward_step_timer()  # hook for step timing

                # post-step hook: gradient clipping, profile gpu, update metrics, count step, step LR scheduler, log
                current_lr = self.optimizer.get_lr()[0]
                self.hook_post_step(
                    step,
                    loss,
                    current_lr,
                    additional_log=additional_log,
                    disable_grad_clip=True,
                    show_log=show_log
                )

            # log train statistics
            loss_per_word = 1.0 * total_loss / n_word_total
            accuracy = 1.0 * n_word_correct / n_word_total
            self.metrics.update_meter(MMeters.TRAIN_LOSS_PER_WORD, loss_per_word)
            self.metrics.update_meter(MMeters.TRAIN_ACC, accuracy)
            
            # return loss_per_word, accuracy
            batch_loss /= num_steps
            batch_snt_loss /= num_steps
            batch_rec_loss /= num_steps
            batch_clip_loss /= num_steps
            
            if self.use_wandb:
                wandb.log({"train_loss": batch_loss})
                wandb.log({"train_snt_loss": batch_snt_loss})
                wandb.log({"train_rec_loss": batch_rec_loss})
                wandb.log({"train_clip_loss": batch_clip_loss})

            # ---------- validation ----------
            do_val = self.check_is_val_epoch()

            is_best = False
            if do_val:
                # run validation including with ground truth tokens and translation without any text
                _val_loss, _val_score, is_best, _metrics = self.validate_epoch(
                    val_loader, datatype=datatype
                )
                print("#############################################")
                print("Do test")
                self.test_epoch(test_loader, datatype=datatype)
                print("###################################################")

            # save the EMA weights
            ema_file = self.exp.get_models_file_ema(self.state.current_epoch)
            torch.save(self.ema.state_dict(), str(ema_file))

            # post-epoch hook: scheduler, save checkpoint, time bookkeeping, feed tensorboard
            self.hook_post_train_and_val_epoch(do_val, is_best)

        # show end of training log message
        self.hook_post_train()
        print("###################################################")
        self.logger.info(
            ", ".join(
                [f"{name} {self.higest_test[name]:.2%}" for name in self.test_metrics]
            )
        )

        # make knn dstore only last epoch
        if make_knn_dstore:
            print("--------------------------")
            print(f'Make Dstore | epoch : {_epoch}')
            print("--------------------------")
            print("---------- Start ----------")
            self.validate_epoch(train_loader, datatype=datatype, make_knn_dstore=make_knn_dstore)
            print('---------- Done -----------')


    @torch.no_grad()
    def validate_epoch(
        self, data_loader: data.DataLoader, 
        datatype: str="bila",
        make_knn_dstore: bool = False,
    ) -> (Tuple[float, float, bool, Dict[str, float]]):
        """
        Run both validation and translation.

        Validation: The same setting as training, where ground-truth word x_{t-1} is used to predict next word x_{t},
        not realistic for real inference.

        Translation: Use greedy generated words to predicted next words, the true inference situation.
        eval_mode can only be set to `val` here, as setting to `test` is cheating
        0. run inference, 1. Get METEOR, BLEU1-4, CIDEr scores, 2. Get vocab size, sentence length

        Args:
            data_loader: Dataloader for validation

        Returns:
            Tuple of:
                validation loss
                validation score
                epoch is best
                custom metrics with translation results dictionary
        """
        self.hook_pre_val_epoch()  # pre val epoch hook: set models to val and start timers
        forward_time_total = 0
        total_loss = 0
        n_word_total = 0
        n_word_correct = 0
        batch_loss = 0.0
        batch_snt_loss = 0.0
        batch_rec_loss = 0.0
        batch_clip_loss = 0.0
        batch_idx = 0

        # setup ema
        if self.ema is not None:
            self.ema.assign(self.model)

        # setup translation submission
        batch_res = {
            "version": "VERSION 1.0",
            "results": defaultdict(list),
            "external_data": {"used": "true", "details": "ay"},
        }
        dataset: BilaDataset = data_loader.dataset

        # ---------- Dataloader Iteration ----------
        num_steps = 0
        pbar = tqdm(
            total=len(data_loader), desc=f"Validate epoch {self.state.current_epoch}"
        )
        for _step, batch in enumerate(data_loader):
            # ---------- forward pass ----------
            self.hook_pre_step_timer()  # hook for step timing

            with autocast(enabled=self.cfg.fp16_val):
                batched_data = [
                    prepare_batch_inputs(
                        step_data,
                        use_cuda=self.cfg.use_cuda,
                        non_blocking=self.cfg.cuda_non_blocking,
                    )
                    for step_data in batch[0]
                ]
                # validate (ground truth as input for next token)
                input_ids_list = [e["input_ids"] for e in batched_data]
                img_feats_list = [e['img_feats'] for e in batched_data]
                txt_feats_list = [e['txt_feats'] for e in batched_data]
                input_masks_list = [e["input_mask"] for e in batched_data]
                token_type_ids_list = [e["token_type_ids"] for e in batched_data]
                input_labels_list = [e["input_labels"] for e in batched_data]
                gt_rec = [e["gt_rec"] for e in batched_data]
                bboxes_list = [e["bboxes"] for e in batched_data]
                bbox_feats_list = [e["bbox_feats"] for e in batched_data]

                loss, pred_scores_list, snt_loss, rec_loss, clip_loss = self.model(
                    input_ids_list,
                    img_feats_list,
                    txt_feats_list,
                    input_masks_list,
                    token_type_ids_list,
                    input_labels_list,
                    gt_rec,
                    bboxes_list,
                    bbox_feats_list,
                )
                batch_loss += loss
                batch_snt_loss += snt_loss
                batch_rec_loss += rec_loss
                batch_clip_loss += clip_loss
                batch_idx += 1
                # translate (no ground truth text)
                step_sizes = batch[1]  # list(int), len == bsz
                meta = batch[2]  # list(dict), len == bsz

                model_inputs = [
                    [e["input_ids"] for e in batched_data],
                    [e["img_feats"] for e in batched_data],
                    [e["txt_feats"] for e in batched_data],
                    [e["input_mask"] for e in batched_data],
                    [e["token_type_ids"] for e in batched_data],
                    [e["bboxes"] for e in batched_data],
                    [e["bbox_feats"] for e in batched_data],
                ]
                dec_seq_list = self.translator.translate_batch(
                    model_inputs,
                    use_beam=self.cfg.use_beam,
                    make_knn_dstore=make_knn_dstore,
                )

                for example_idx, (step_size, cur_meta) in enumerate(
                    zip(step_sizes, meta)
                ):
                    # example_idx indicates which example is in the batch
                    for step_idx, step_batch in enumerate(dec_seq_list[:step_size]):
                        # step_idx or we can also call it sen_idx
                        batch_res["results"][cur_meta["clip_id"]].append(
                            {
                                "sentence": dataset.convert_ids_to_sentence(
                                    step_batch[example_idx].cpu().tolist()
                                ),
                                # remove encoding
                                # .encode("ascii", "ignore"),
                                "gt_sentence": cur_meta["gt_sentence"],
                                "clip_id": cur_meta["clip_id"]
                            }
                        )

                # keep logs
                n_correct = 0
                n_word = 0
                for pred, gold in zip(pred_scores_list, input_labels_list):
                    n_correct += cal_performance(pred, gold)
                    valid_label_mask = gold.ne(BilaDataset.IGNORE)
                    n_word += valid_label_mask.sum().item()

                # calculate metrix
                n_word_total += n_word
                n_word_correct += n_correct
                total_loss += loss.item()

            # end of step
            self.hook_post_forward_step_timer()
            forward_time_total += self.timedelta_step_forward
            num_steps += 1

            if self.cfg.debug:
                break

            pbar.update()
        pbar.close()

        batch_loss /= batch_idx
        batch_snt_loss /= batch_idx
        batch_rec_loss /= batch_idx
        batch_clip_loss /= batch_idx
        loss_delta = self.beforeloss - batch_loss
        if self.use_wandb and not make_knn_dstore:
            wandb.log({"val_loss_diff": loss_delta})
            wandb.log({"val_loss": batch_loss})
            wandb.log({"val_snt_loss": batch_snt_loss})
            wandb.log({"val_rec_loss": batch_rec_loss})
            wandb.log({"val_clip_loss": batch_clip_loss})
        self.beforeloss = batch_loss

        # sort translation
        batch_res["results"] = self.translator.sort_res(batch_res["results"])

        # write translation results of this epoch to file
        if make_knn_dstore:
            eval_mode = "train"  # which dataset split
        else:
            eval_mode = self.cfg.dataset_val.split  # which dataset split

        file_translation_raw = self.exp.get_translation_files(
            self.state.current_epoch, eval_mode, make_knn_dstore=make_knn_dstore
        )
        if datatype == 'bila':
            json.dump(batch_res, file_translation_raw.open("wt", encoding="utf8"))
        elif datatype == 'bilas':
            json.dump(batch_res, file_translation_raw.open("wt", encoding="utf8"), ensure_ascii=False)
        

        # get reference files (ground truth captions)
        reference_files_map = get_reference_files(
            self.cfg.dataset_val.name, self.exp.data_dir, datatype=datatype, make_knn_dstore=make_knn_dstore,
        )
        reference_files = reference_files_map[eval_mode]
        reference_file_single = reference_files[0]
        
        if make_knn_dstore:
            print('----------------------------')
            print(file_translation_raw)
            print(reference_files)
            print('----------------------------')

        # language evaluation
        res_lang = evaluate_language_files(
            file_translation_raw, reference_files, verbose=False, all_scorer=True, datatype=datatype
        )
        # basic stats
        res_stats = evaluate_stats_files(
            file_translation_raw, reference_file_single, verbose=False
        )
        # repetition
        res_rep = evaluate_repetition_files(
            file_translation_raw, reference_file_single, verbose=False, datatype=datatype,
        )

        # merge results
        all_metrics = {**res_lang, **res_stats, **res_rep}
        assert len(all_metrics) == len(res_lang) + len(res_stats) + len(
            res_rep
        ), "Lost infos while merging translation results!"

        # flatten results and make them json compatible
        flat_metrics = {}
        for key, val in all_metrics.items():
            if isinstance(val, Mapping):
                for subkey, subval in val.items():
                    flat_metrics[f"{key}_{subkey}"] = subval
                continue
            flat_metrics[key] = val
        for key, val in flat_metrics.items():
            if isinstance(val, (np.float16, np.float32, np.float64)):
                flat_metrics[key] = float(val)

        # feed meters
        for result_key, meter_name in TRANSLATION_METRICS.items():
            self.metrics.update_meter(meter_name, flat_metrics[result_key])

        # log translation results
        self.logger.info(
            f"Done with translation, epoch {self.state.current_epoch} split {eval_mode}"
        )
        self.logger.info(
            ", ".join(
                [f"{name} {flat_metrics[name]:.2%}" for name in TRANSLATION_METRICS_LOG]
            )
        )

        # calculate and output validation metrics
        loss_per_word = 1.0 * total_loss / n_word_total
        accuracy = 1.0 * n_word_correct / n_word_total
        self.metrics.update_meter(MMeters.TRAIN_LOSS_PER_WORD, loss_per_word)
        self.metrics.update_meter(MMeters.TRAIN_ACC, accuracy)
        forward_time_total /= num_steps
        self.logger.info(
            f"Loss {loss_per_word:.5f} Acc {accuracy:.3%} total {timer() - self.timer_val_epoch:.3f}s, "
            f"forward {forward_time_total:.3f}s"
        )

        # find field which determines whether this is a new best epoch
        if self.use_wandb:
            wandb.log({"val_BLEU4": flat_metrics["Bleu_4"], "val_METEOR": flat_metrics["METEOR"], "val_ROUGE_L": flat_metrics["ROUGE_L"], "val_CIDEr": flat_metrics["CIDEr"]})
        if self.cfg.val.det_best_field == "cider":
            # val_score = flat_metrics["CIDEr"]
            val_score = -1 * batch_loss
        else:
            raise NotImplementedError(
                f"best field {self.cfg.val.det_best_field} not known"
            )

        # check for a new best epoch and update validation results
        is_best = self.check_is_new_best(val_score)
        if is_best == True:
            self.best_epoch = self.state.current_epoch
        self.hook_post_val_epoch(loss_per_word, is_best)

        if self.is_test:
            # for test runs, save the validation results separately to a file
            self.metrics.feed_metrics(
                False, self.state.total_step, self.state.current_epoch
            )
            metrics_file = (
                self.exp.path_base / f"val_ep_{self.state.current_epoch}.json"
            )
            self.metrics.save_epoch_to_file(metrics_file)
            self.logger.info(f"Saved validation results to {metrics_file}")

            # update the meteor metric in the result if it's -999 because java crashed. only in some conditions
            best_ep = self.exp.find_best_epoch()
            self.logger.info(
                f"Dataset split config {self.cfg.dataset_val.split} loaded {self.load_ep} best {best_ep}"
            )
            if (
                self.cfg.dataset_val.split == "val"
                and self.load_ep == best_ep == self.state.current_epoch
            ):
                # load metrics file and write it back with the new meteor IFF meteor is -999
                metrics_file = self.exp.get_metrics_epoch_file(best_ep)
                metrics_data = json.load(metrics_file.open("rt", encoding="utf8"))
                # metrics has stored meteor as a list of tuples (epoch, value). convert to dict, update, convert back.
                meteor_dict = dict(metrics_data[TextMetricsConst.METEOR])
                if ((meteor_dict[best_ep] + 999) ** 2) < 1e-4:
                    meteor_dict[best_ep] = flat_metrics[TextMetricsConstEvalCap.METEOR]
                    metrics_data[TextMetricsConst.METEOR] = list(meteor_dict.items())
                    json.dump(metrics_data, metrics_file.open("wt", encoding="utf8"))
                    self.logger.info(f"Updated meteor in file {metrics_file}")

        return total_loss, val_score, is_best, flat_metrics


    @torch.no_grad()
    def test_epoch(
        self,
        data_loader: data.DataLoader,
        datatype: str = 'bila',
    ) -> (Tuple[float, float, bool, Dict[str, float]]):
        """
        Run both validation and translation.

        Validation: The same setting as training, where ground-truth word x_{t-1} is used to predict next word x_{t},
        not realistic for real inference.

        Translation: Use greedy generated words to predicted next words, the true inference situation.
        eval_mode can only be set to `val` here, as setting to `test` is cheating
        0. run inference, 1. Get METEOR, BLEU1-4, CIDEr scores, 2. Get vocab size, sentence length

        Args:
            data_loader: Dataloader for validation

        Returns:
            Tuple of:
                validation loss
                validation score
                epoch is best
                custom metrics with translation results dictionary
        """
        self.hook_pre_val_epoch()  # pre val epoch hook: set models to val and start timers
        forward_time_total = 0
        total_loss = 0
        n_word_total = 0
        n_word_correct = 0

        # setup ema
        if self.ema is not None:
            self.ema.assign(self.model)

        # setup translation submission
        batch_res = {
            "version": "VERSION 1.0",
            "results": defaultdict(list),
            "external_data": {"used": "true", "details": "ay"},
        }
        dataset: BilaDataset = data_loader.dataset

        # ---------- Dataloader Iteration ----------
        num_steps = 0
        pbar = tqdm(
            total=len(data_loader), desc=f"Validate epoch {self.state.current_epoch}"
        )
        batch_loss = 0.0
        batch_snt_loss = 0.0
        batch_rec_loss = 0.0
        batch_clip_loss = 0.0
        batch_idx = 0
        for _step, batch in enumerate(data_loader):
            # ---------- forward pass ----------
            self.hook_pre_step_timer()  # hook for step timing

            with autocast(enabled=self.cfg.fp16_val):
                batched_data = [
                    prepare_batch_inputs(
                        step_data,
                        use_cuda=self.cfg.use_cuda,
                        non_blocking=self.cfg.cuda_non_blocking,
                    )
                    for step_data in batch[0]
                ]
                # validate (ground truth as input for next token)
                input_ids_list = [e["input_ids"] for e in batched_data]
                img_feats_list = [e["img_feats"] for e in batched_data]
                txt_feats_list = [e["txt_feats"] for e in batched_data]
                input_masks_list = [e["input_mask"] for e in batched_data]
                token_type_ids_list = [e["token_type_ids"] for e in batched_data]
                input_labels_list = [e["input_labels"] for e in batched_data]
                gt_rec = [e["gt_rec"] for e in batched_data]
                bboxes_list = [e["bboxes"] for e in batched_data]
                bbox_feats_list = [e["bbox_feats"] for e in batched_data]

                loss, pred_scores_list, snt_loss, rec_loss, clip_loss = self.model(
                    input_ids_list,
                    img_feats_list,
                    txt_feats_list,
                    input_masks_list,
                    token_type_ids_list,
                    input_labels_list,
                    gt_rec,
                    bboxes_list,
                    bbox_feats_list,
                )
                batch_loss += loss
                batch_snt_loss += snt_loss
                batch_rec_loss += rec_loss
                batch_clip_loss += clip_loss
                batch_idx += 1
                # translate (no ground truth text)
                step_sizes = batch[1]  # list(int), len == bsz
                meta = batch[2]  # list(dict), len == bsz

                model_inputs = [
                    [e["input_ids"] for e in batched_data],
                    [e["img_feats"] for e in batched_data],
                    [e["txt_feats"] for e in batched_data],
                    [e["input_mask"] for e in batched_data],
                    [e["token_type_ids"] for e in batched_data],
                    [e["bboxes"] for e in batched_data],
                    [e["bbox_feats"] for e in batched_data],
                ]
                dec_seq_list = self.translator.translate_batch(
                    model_inputs,
                    use_beam=self.cfg.use_beam,
                )

                for example_idx, (step_size, cur_meta) in enumerate(
                    zip(step_sizes, meta)
                ):
                    # example_idx indicates which example is in the batch
                    for step_idx, step_batch in enumerate(dec_seq_list[:step_size]):
                        # step_idx or we can also call it sen_idx
                        batch_res["results"][cur_meta["clip_id"]].append(
                            {
                                "sentence": dataset.convert_ids_to_sentence(
                                    step_batch[example_idx].cpu().tolist()
                                ),
                                "gt_sentence": cur_meta["gt_sentence"],
                                "clip_id": cur_meta["clip_id"]
                            }
                        )

                # keep logs
                n_correct = 0
                n_word = 0
                for pred, gold in zip(pred_scores_list, input_labels_list):
                    n_correct += cal_performance(pred, gold)
                    valid_label_mask = gold.ne(BilaDataset.IGNORE)
                    n_word += valid_label_mask.sum().item()

                # calculate metrix
                n_word_total += n_word
                n_word_correct += n_correct
                total_loss += loss.item()

            # end of step
            self.hook_post_forward_step_timer()
            forward_time_total += self.timedelta_step_forward
            num_steps += 1

            if self.cfg.debug:
                break

            pbar.update()
        pbar.close()
        batch_loss /= batch_idx
        batch_snt_loss /= batch_idx
        batch_rec_loss /= batch_idx
        batch_clip_loss /= batch_idx
        if self.use_wandb:
            wandb.log({"test_loss": batch_loss})
            wandb.log({"test_snt_loss": batch_snt_loss})
            wandb.log({"test_rec_loss": batch_rec_loss})
            wandb.log({"test_clip_loss": batch_clip_loss})

        # ---------- validation done ----------

        # sort translation
        batch_res["results"] = self.translator.sort_res(batch_res["results"])

        # write translation results of this epoch to file
        eval_mode = "test"
        file_translation_raw = self.exp.get_translation_files(
            self.state.current_epoch, "test"
        )
        if datatype == "bila":
            json.dump(batch_res, file_translation_raw.open("wt", encoding="utf8"))    
        elif datatype == 'bilas':
            json.dump(batch_res, file_translation_raw.open("wt", encoding="utf8"), ensure_ascii=False)
        
        # get reference files (ground truth captions)
        reference_files_map = get_reference_files(
            self.cfg.dataset_val.name, self.exp.data_dir, test=True, datatype=datatype,
        )
        reference_files = reference_files_map[eval_mode]
        reference_file_single = reference_files[0]

        # language evaluation
        res_lang = evaluate_language_files(
            file_translation_raw, reference_files, verbose=False, all_scorer=True, datatype=datatype
        )
        # basic stats
        res_stats = evaluate_stats_files(
            file_translation_raw, reference_file_single, verbose=False
        )
        # repetition
        res_rep = evaluate_repetition_files(
            file_translation_raw, reference_file_single, verbose=False, datatype=datatype,
        )

        # merge results
        all_metrics = {**res_lang, **res_stats, **res_rep}
        assert len(all_metrics) == len(res_lang) + len(res_stats) + len(
            res_rep
        ), "Lost infos while merging translation results!"

        # flatten results and make them json compatible
        flat_metrics = {}
        for key, val in all_metrics.items():
            if isinstance(val, Mapping):
                for subkey, subval in val.items():
                    flat_metrics[f"{key}_{subkey}"] = subval
                continue
            flat_metrics[key] = val
        for key, val in flat_metrics.items():
            if isinstance(val, (np.float16, np.float32, np.float64)):
                flat_metrics[key] = float(val)

        # feed meters
        for result_key, meter_name in TRANSLATION_METRICS.items():
            self.metrics.update_meter(meter_name, flat_metrics[result_key])

        # log translation results
        self.logger.info(
            f"Done with translation, epoch {self.state.current_epoch} split {eval_mode}"
        )
        if self.use_wandb:
            wandb.log({"test_BLEU4": flat_metrics["Bleu_4"], "test_METEOR": flat_metrics["METEOR"], "test_ROUGE_L": flat_metrics["ROUGE_L"], "test_CIDEr": flat_metrics["CIDEr"]})
        
        self.test_metrics = TRANSLATION_METRICS_LOG
        self.higest_test = flat_metrics
        self.logger.info(
            ", ".join(
                [f"{name} {flat_metrics[name]:.2%}" for name in TRANSLATION_METRICS_LOG]
            )
        )


    def get_opt_state(self) -> Dict[str, Dict[str, nn.Parameter]]:
        """
        Return the current optimizer and scheduler state.
        Note that the BertAdam optimizer used already includes scheduling.

        Returns:
            Dictionary of optimizer and scheduler state dict.
        """
        return {
            "optimizer": self.optimizer.state_dict()
            # "lr_scheduler": self.lr_scheduler.state_dict()
        }

    def set_opt_state(self, opt_state: Dict[str, Dict[str, nn.Parameter]]) -> None:
        """
        Set the current optimizer and scheduler state from the given state.

        Args:
            opt_state: Dictionary of optimizer and scheduler state dict.
        """
        self.optimizer.load_state_dict(opt_state["optimizer"])
        # self.lr_scheduler.load_state_dict(opt_state["lr_scheduler"])

    def get_files_for_cleanup(self, epoch: int) -> List[Path]:
        """
        Implement this in the child trainer.

        Returns:
            List of files to cleanup.
        """
        return [
            # self.exp.get_translation_files(epoch, split="train"),
            self.exp.get_translation_files(epoch, split="val"),
            self.exp.get_models_file_ema(epoch),
        ]

    def check_is_val_epoch(self) -> bool:
        """
        Check if validation is needed at the end of training epochs.

        Returns:
            Whether or not validation is needed.
        """
        # check if we need to validate
        do_val = (self.state.current_epoch % self.cfg.val.val_freq == 0 and self.cfg.val.val_freq > -1
                  and self.state.current_epoch >= self.cfg.val.val_start)
        # always validate the last epoch
        do_val = do_val or self.state.current_epoch == self.cfg.train.num_epochs
        return do_val

    def check_is_new_best(self, result: float) -> bool:
        """
        Check if the given result improves over the old best.

        Args:
            result: Validation result to compare with old best.

        Returns:
            Whether or not the result improves over the old best.
        """
        old_best = self.state.det_best_field_best

        # check if this is a new best
        is_best = self._check_if_current_score_is_best(result, old_best)

        # log info
        old_best_str = f"{old_best:.5f}" if old_best is not None else "NONE"
        self.logger.info(f"***** Improvement: {is_best} *****. Before: {old_best_str}, "
                         f"After {result:.5f}, Field: {self.cfg.val.det_best_field}, "
                         f"Mode {self.cfg.val.det_best_threshold_mode}")

        # update fields
        self.state.det_best_field_current = result
        if is_best:
            self.state.det_best_field_best = result

        return is_best

    def close(self) -> None:
        """
        Close logger and metric writers.
        """
        utils.remove_handlers_from_logger(self.logger)
        self.metrics.close()

    # ---------- Public hooks that run once per experiment ----------

    def hook_post_init(self) -> None:
        """
        Hook called after trainer init is done. Loads the correct epoch.
        """
        if self.load:
            assert not self.model_mgr.was_loaded, (
                f"Error: Loading epoch {self.load_ep} but already weights have been loaded. If you load weights for "
                f"warmstarting, you cannot run if the experiments has already saved checkpoints. Change the run name "
                f"or use --reset to delete the experiment run.")
            if self.load_model:
                # load model from file. this would start training from epoch 0, but is usually only used for validation.
                self.logger.info(f"Loading model from checkpoint file {self.load_model}")
                model_state = torch.load(str(self.load_model))
                self.model_mgr.set_model_state(model_state)
            else:
                # load model given an epoch. also reload metrics and optimization to correctly continue training.
                self.logger.info(f"Loading Ep {self.load_ep}.")
                self._load_checkpoint(self.load_ep)
                if not self.is_test:
                    # In training, add 1 to current epoch after loading since if we loaded epoch N, we are training
                    # epoch N+1 now. In validation, we are validating on epoch N.
                    self.state.current_epoch += 1

    def hook_pre_train(self, show_log:bool=False) -> None:
        """
        Hook called on training start. Remember start epoch, time the start, log info.
        """
        self.state.start_epoch = self.state.current_epoch
        self.timer_train_start = timer()
        if show_log:
            self.logger.info(f"Training from {self.state.current_epoch} to {self.cfg.train.num_epochs}")
            self.logger.info("Training Models on devices " + ", ".join([
                f"{key}: {next(val.parameters()).device}" for key, val in self.model_mgr.model_dict.items()]))

    def hook_post_train(self, show_log:bool=False) -> None:
        """
        Hook called on training finish. Log info on total num epochs trained and duration.
        """
        if show_log:
            self.logger.info(f"In total, training {self.state.current_epoch} epochs took "
                            f"{self.state.time_total:.3f}s ({self.state.time_total - self.state.time_val:.3f}s "
                            f"train / {self.state.time_val:.3f}s val)")

    # ---------- Public hooks that run every epoch ----------

    def hook_pre_train_epoch(self, show_log:bool=False) -> None:
        """
        Hook called before training an epoch. Set models to train, times start, reset meters, log info.
        """
        self.model_mgr.set_all_models_train()
        self.timer_train_epoch = timer()
        self.timer_step = timer()
        # clear metrics
        self.metrics.hook_epoch_start()
        if show_log:
            self.logger.info(f"{str(datetime.datetime.now()).split('.')[0]} ---------- "
                            f"Training epoch: {self.state.current_epoch}")

    def hook_pre_val_epoch(self) -> None:
        """
        Hook called before validating an epoch. Set models to val, times start.
        """
        # set models to validation mode
        self.model_mgr.set_all_models_eval()
        # start validation epoch timer
        self.timer_val_epoch = timer()
        #
        self.timer_step = timer()

    def hook_post_val_epoch(self, val_loss: float, is_best: bool) -> None:
        """
        Hook called after validation epoch is done. Updates basic validation meters.

        Args:
            val_loss: Validation loss.
            is_best: Whether this is a new best epoch.
        """
        # update validation timer
        self.state.time_val += timer() - self.timer_val_epoch

        # update loss and result
        self.metrics.update_meter(Metrics.VAL_LOSS, val_loss)
        self.metrics.update_meter(Metrics.VAL_BEST_FIELD, self.state.det_best_field_current)

        # update info dict for reloading
        self.state.infos_val_epochs.append(self.state.current_epoch)
        self.state.infos_val_steps.append(self.state.total_step)
        self.state.infos_val_is_good.append(is_best)

    def hook_post_train_and_val_epoch(self, is_val: bool, has_improved: bool) -> None:
        """
        Hook called after entire epoch (training + validation) is done.

        Args:
            is_val: Whether there was validation done this epoch.
            has_improved: If there was validation, whether there was an improvement (new best).
        """
        # update total timer
        self.state.time_total += timer() - self.timer_train_epoch

        # step LR scheduler after end of epoch
        if self.lr_scheduler is not None:
            self.lr_scheduler.step_epoch(is_val, has_improved)

        # log metrics
        self.metrics.update_meter(Metrics.TIME_TOTAL, self.state.time_total)
        self.metrics.update_meter(Metrics.TIME_VAL, self.state.time_val)
        self.metrics.update_meter(Metrics.TRAIN_EPOCH, self.state.current_epoch)

        # display step times
        fields = [Metrics.TIME_STEP_FORWARD, Metrics.TIME_STEP_BACKWARD, Metrics.TIME_STEP_OTHER]
        time_total = self.metrics.meters[Metrics.TIME_STEP_TOTAL].avg
        time_str_list = ["Step time: Total", f"{time_total * 1000:.0f}ms"]
        for field in fields:
            time_value = self.metrics.meters[field].avg
            time_name_short = str(field).split("/")[-1].split("_")[-1]
            time_str_list += [time_name_short, f"{time_value * 1000:.2f}ms", f"{time_value / time_total:.1%}"]
        self.logger.info(" ".join(time_str_list))

        # feed step-based metrics to tensorboard and collector
        self.metrics.feed_metrics(False, self.state.total_step, self.state.current_epoch)

        # save checkpoint and metrics
        self._save_checkpoint()

        # cleanup files depending on saving config (default just keep best and last epoch, discard all others)
        self._cleanup_files()

        # increase epoch counter
        self.state.current_epoch += 1

    # ---------- Public hooks that run every step ----------

    def hook_pre_step_timer(self) -> None:
        """
        Hook called before forward pass. Sets timer.
        """
        self.timer_step_forward = timer()

    def hook_post_forward_step_timer(self) -> None:
        """
        Hook called after forward pass, before backward pass. Compute time delta and sets timer.
        """
        self.timer_step_backward = timer()
        self.timedelta_step_forward = self.timer_step_backward - self.timer_step_forward

    def hook_post_backward_step_timer(self) -> None:
        """
        Hook called after backward pass. Compute time delta.
        """
        self.timedelta_step_backward = timer() - self.timer_step_backward

    def hook_post_step(
            self, epoch_step: int, loss: torch.Tensor, lr: float, additional_log: Optional[str] = None,
            disable_grad_clip: bool = False, show_log: bool=False) -> bool:
        """
        Hook called after one optimization step.

        Profile gpu and update step-based meters. Feed everything to tensorboard.
        Needs some information to be passed down from the trainer for proper logging.

        Args:
            epoch_step: Current step in the epoch.
            loss: Training loss.
            lr: Training learning rate.
            additional_log: Additional string to print in the train step log.
            disable_grad_clip: Disable gradient clipping if it's done already somewhere else

        Returns:
            Whether log output should be printed in this step or not.
        """
        # compute total time for this step and restart the timer
        total_step_time = timer() - self.timer_step
        self.timer_step = timer()

        # clip gradients
        total_norm = 0
        if self.cfg.train.clip_gradient > -1 and not disable_grad_clip:
            # get all parameters to clip
            _params, _param_names, params_flat = self.model_mgr.get_all_params()
            # clip using pytorch
            total_norm = clip_grad_norm_(params_flat, self.cfg.train.clip_gradient)
            if total_norm > self.cfg.train.clip_gradient:
                # print log message if gradients where clipped
                grad_clip_coef = self.cfg.train.clip_gradient / (total_norm + 1e-6)
                self.logger.info(f"Clipping gradient: {total_norm} with coef {grad_clip_coef}")
            total_norm = total_norm.item()
        self.state.last_grad_norm = total_norm

        # print infos
        if epoch_step % self.cfg.logging.step_train == 0:
            total_train_time = (timer() - self.timer_train_epoch) / 60
            str_step = ("{:" + str(len(str(self.steps_per_epoch))) + "d}").format(epoch_step)
            print_string = "".join([
                f"E{self.state.current_epoch}[{str_step}/{self.steps_per_epoch}] T {total_train_time:.3f}m ",
                f"LR {lr:.1e} L {loss:.4f} ",
                f"Grad {self.state.last_grad_norm:.3e} " if self.state.last_grad_norm != 0 else "",
                f"{additional_log}" if additional_log is not None else ""])
            if show_log:
                self.logger.info(print_string)

        # check GPU / RAM profiling
        if ((self.state.epoch_step % self.cfg.logging.step_gpu == 0 and self.cfg.logging.step_gpu > 0) or
                self.state.epoch_step == self.cfg.logging.step_gpu_once and self.cfg.logging.step_gpu_once > 0):
            # get the current profile values
            (gpu_names, total_memory_per, used_memory_per, load_per, ram_total, ram_used, ram_avail
            ) = utils.profile_gpu_and_ram()
            # average / sum over all GPUs
            gpu_mem_used: float = sum(used_memory_per)
            gpu_mem_total: float = sum(total_memory_per)
            # gpu_mem_percent: float = gpu_mem_used / gpu_mem_total
            load_avg: float = sum(load_per) / max(1, len(load_per))

            self.metrics.update_meter(Metrics.PROFILE_GPU_MEM_USED, gpu_mem_used)
            self.metrics.update_meter(Metrics.PROFILE_GPU_MEM_TOTAL, gpu_mem_total)
            self.metrics.update_meter(Metrics.PROFILE_GPU_LOAD, load_avg)
            self.metrics.update_meter(Metrics.PROFILE_RAM_USED, ram_used)
            self.metrics.update_meter(Metrics.PROFILE_RAM_TOTAL, ram_total)
            # # these 2 are not logged as they are redundant with the others.
            # self.metrics.update_meter(Metrics.PROFILE_GPU_MEM_PERCENT, gpu_mem_percent)
            # self.metrics.update_meter(Metrics.PROFILE_RAM_AVAILABLE, ram_avail)

            # log the values
            gpu_names_str = " ".join(set(gpu_names))
            multi_load, multi_mem = "", ""
            if len(load_per) > 1:
                multi_load = " [" + ", ".join(f"{load:.0%}" for load in load_per) + "]"
                multi_mem = " [" + ", ".join(f"{mem:.1f}GB" for mem in used_memory_per) + "]"
            if show_log:
                self.logger.info(f"RAM GB used/avail/total: {ram_used:.1f}/{ram_avail:.1f}/{ram_total:.1f} - "
                                f"GPU {gpu_names_str} Load: {load_avg:.1%}{multi_load} "
                                f"Mem: {gpu_mem_used:.1f}GB/{gpu_mem_total:.1f}GB{multi_mem}")

        # update timings
        other_t = total_step_time - self.timedelta_step_forward - self.timedelta_step_backward
        self.metrics.update_meter(Metrics.TIME_STEP_FORWARD, self.timedelta_step_forward)
        self.metrics.update_meter(Metrics.TIME_STEP_BACKWARD, self.timedelta_step_backward)
        self.metrics.update_meter(Metrics.TIME_STEP_TOTAL, total_step_time)
        self.metrics.update_meter(Metrics.TIME_STEP_OTHER, other_t)
        # update clipped gradient
        self.metrics.update_meter(Metrics.TRAIN_GRAD_CLIP, self.state.last_grad_norm)
        # update LR
        self.metrics.update_meter(Metrics.TRAIN_LR, lr)
        if self.state.epoch_step % self.cfg.logging.step_train == 0 and self.cfg.logging.step_train > 0:
            # loss update necessary
            self.metrics.update_meter(Metrics.TRAIN_LOSS, loss.item())

        # Save epoch step and increase total step counter
        self.state.epoch_step = epoch_step
        self.state.total_step += 1

        # feed step-based metrics to tensorboard and collector
        self.metrics.feed_metrics(True, self.state.total_step, self.state.current_epoch)

        # End of batch, step lr scheduler depending on flag
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    # ---------- Non-public methods ----------

    def _check_if_current_score_is_best(self, current: float, best: float) -> bool:
        """
        Compare given current and best and return True if the current is better than best + some threshold.
        Depending on config, smaller or bigger values are better and threshold is absolute or relative.

        Args:
            current: Current score.
            best: Best score so far.

        Returns:
            Whether current is better than best by some threshold.
        """
        cp_mode = self.cfg.val.det_best_compare_mode
        th_mode = self.cfg.val.det_best_threshold_mode

        if best is None:
            # no best exists, so current is automatically better
            return True
        if cp_mode == MetricComparisonConst.VAL_DET_BEST_MODE_MIN:
            # smaller values are better
            if th_mode == MetricComparisonConst.VAL_DET_BEST_TH_MODE_REL:
                # must be relatively better by epsilon
                rel_epsilon = 1 - self.cfg.val.det_best_threshold_value
                return current < best * rel_epsilon
            if th_mode == MetricComparisonConst.VAL_DET_BEST_TH_MODE_ABS:
                # must be absolutely better by epsilon
                return current < best - self.cfg.val.det_best_threshold_value
            raise ValueError(f"Threshold mode for metric comparison not understood: {th_mode}")
        if cp_mode == MetricComparisonConst.VAL_DET_BEST_MODE_MAX:
            # bigger values are better
            if th_mode == MetricComparisonConst.VAL_DET_BEST_TH_MODE_REL:
                # must be relatively better by epsilon
                rel_epsilon = 1 + self.cfg.val.det_best_threshold_value
                return current > best * rel_epsilon
            if th_mode == MetricComparisonConst.VAL_DET_BEST_TH_MODE_ABS:
                # must be absolutely better by epsilon
                return current > best + self.cfg.val.det_best_threshold_value
            raise ValueError(f"Threshold mode for metric comparison not understood: {th_mode}")
        raise ValueError(f"Compare mode for determining best field not understood: {cp_mode}")

    def _save_checkpoint(self) -> None:
        """
        Save current epoch.
        """
        # trainer state
        trainerstate_file = self.exp.get_trainerstate_file(self.state.current_epoch)
        self.state.save(trainerstate_file)

        # metrics state
        self.metrics.save_epoch(self.state.current_epoch)

        # models
        models_file = self.exp.get_models_file(self.state.current_epoch)
        state = self.model_mgr.get_model_state()
        torch.save(state, str(models_file))

        # optimizer and scheduler
        opt_file = self.exp.get_optimizer_file(self.state.current_epoch)
        opt_state = self.get_opt_state()
        torch.save(opt_state, str(opt_file))

    def _load_checkpoint(self, epoch) -> None:
        """
        Load given epoch.
        """
        # trainer state
        trainerstate_file = self.exp.get_trainerstate_file(epoch)
        self.state.load(trainerstate_file)

        # metrics state
        self.metrics.load_epoch(epoch)

        # models
        models_file = self.exp.get_models_file(epoch)
        model_state = torch.load(str(models_file))
        self.model_mgr.set_model_state(model_state)

        # optimizer and scheduler
        if not self.is_test:
            opt_file = self.exp.get_optimizer_file(self.state.current_epoch)
            opt_state = torch.load(str(opt_file))
            self.set_opt_state(opt_state)
        else:
            self.logger.info("Don't load optimizer and scheduler during inference.")

    def _cleanup_files(self) -> None:
        """
        Delete epoch and info files to save space, depending on configuration.
        """
        ep_nums = self.exp.get_existing_checkpoints()
        if len(ep_nums) == 0:
            # no checkpoints exist
            return
        # always save best and last
        best_ep = self.exp.find_best_epoch()
        last_ep = ep_nums[-1]
        # remember which epochs have been cleaned up
        cleaned = []
        for ep_num in ep_nums:
            # always keep the best episode
            if ep_num == best_ep:
                continue
            # always keep the last episode
            if ep_num == last_ep:
                continue
            # if the save checkpoint frequency is set, some intermediate checkpoints should be kept
            if self.cfg.saving.keep_freq > 0:
                if ep_num % self.cfg.saving.keep_freq == 0:
                    continue
            # delete safely (don't crash if they don't exist for some reason)
            for file in [self.exp.get_models_file(ep_num), self.exp.get_optimizer_file(ep_num),
                         self.exp.get_trainerstate_file(ep_num), self.exp.get_metrics_epoch_file(ep_num),
                         self.exp.get_metrics_step_file(ep_num)] + self.get_files_for_cleanup(ep_num):
                if file.is_file():
                    os.remove(file)
                else:
                    self.logger.warning(f"Tried to delete {file} but couldn't find it.")
            cleaned.append(ep_num)
        if len(cleaned) > 0:
            self.logger.debug(f"Deleted epochs: {cleaned}")