"""
Definition of constants and configurations for captioning with MART.
"""

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from utils.baseconfig import ConfigClass, ConstantHolder
from utils import baseconfig, utils
from utils import utils


class DatasetConfig(ConfigClass):
    """
    MART Dataset Configuration class

    Args:
        config: Configuration dictionary to be loaded, dataset part.
    """

    def __init__(self, config: Dict) -> None:
        # general dataset info
        self.name: str = config.pop("name")
        self.split: str = config.pop("split")
        
        # general dataloader configuration
        self.shuffle: bool = config.pop("shuffle")
        self.pin_memory: bool = config.pop("pin_memory")
        self.num_workers: int = config.pop("num_workers")


class Config(ConfigClass):
    """
    Args:
        config: Configuration dictionary to be loaded, logging part.

    Attributes:
        train: Training config
        val: Validation config
        dataset_train: Train dataset config
        dataset_val: Val dataset config
        logging: Log frequency

        max_n_sen: for recurrent, max number of sentences, 6 for anet, 12 for yc2
        max_n_sen_add_val: Increase max_n_sen during validation (default 10)
        max_t_len: max length of text (sentence or paragraph), 30 for anet, 20 for yc2
        max_v_len: max length of video feature (default 100)
        type_vocab_size: Size of sequence type vocabulary: video as 0, text as 1 (default 2)
        word_vec_size: GloVE embeddings size (default 300)

        debug: Activate debugging. Unused / untested (default False)

        attention_probs_dropout_prob: Dropout on attention mask (default 0.1)
        hidden_dropout_prob: Dropout on hidden states (default 0.1)
        hidden_size: Model hidden size (default 768)
        intermediate_size: Model intermediate size (default 768)
        layer_norm_eps: Epsilon parameter for Layernorm (default 1e-12)
        max_position_embeddings: Position embeddings limit (default 25)
        memory_dropout_prob: Dropout on memory cells (default 0.1)
        num_attention_heads: Number of attention heads (default 12)
        num_hidden_layers: number of transformer layers (default 2)
        n_memory_cells: number of memory cells in each layer (default 1)
        share_wd_cls_weight: share weight matrix of the word embedding with the final classifier (default False)
        use_glove: Disable loading GloVE embeddings. (default None)
        freeze_glove: do not train GloVe vectors (default False)
        model_type: This is inferred from the fields recurrent, untied, mtrans, xl, xl_grad

        label_smoothing: Use soft target instead of one-hot hard target (default 0.1)

        save_mode: all: save models at each epoch. best: only save the best model (default best, choices: all, best)
        use_beam: use beam search, otherwise greedy search (default False)
        beam_size: beam size (default 2)
        n_best: stop searching when get n_best from beam search (default 1)

        ema_decay: Use exponential moving average at training, float in (0, 1) and -1: do not use.
            ema_param = new_param * ema_decay + (1-ema_decay) * last_param (default 0.9999)
        initializer_range: Weight initializer range (default 0.02)
        lr: Learning rate (default 0.0001)
        lr_warmup_proportion: Proportion of training to perform linear learning rate warmup for.
            E.g., 0.1 = 10%% of training. (default 0.1)
    """

    def __init__(self, config: Dict[str, Any], strict: bool=True) -> None:
        # basic setting
        self.description: str = config.pop("description", "no description given.")
        self.strict = strict
        self.config = config
        self.config_orig = deepcopy(config)
        
        # "same_as" -> dict
        utils.resolve_sameas_config_recursively(config)

        # mandatory groups, needed for nntrainer to work correctly
        self.train = TrainConfig(config.pop("train"))
        self.val = ValConfig(config.pop("val"))
        self.dataset_train = DatasetConfig(config.pop("dataset_train"))
        self.dataset_val = DatasetConfig(config.pop("dataset_val"))
        self.logging = LoggingConfig(config.pop("logging"))
        self.saving = SavingConfig(config.pop("saving"))
        
        # more training
        self.label_smoothing: float = config.pop("label_smoothing")
        
        # more validation
        self.save_mode: str = config.pop("save_mode")
        self.use_beam: bool = config.pop("use_beam")
        self.beam_size: int = config.pop("beam_size")
        self.n_best: int = config.pop("n_best")
        self.min_sen_len: int = config.pop("min_sen_len")
        self.max_sen_len: int = config.pop("max_sen_len")
        self.block_ngram_repeat: int = config.pop("block_ngram_repeat")
        self.length_penalty_name: str = config.pop("length_penalty_name")
        self.length_penalty_alpha: float = config.pop("length_penalty_alpha")

        # dataset
        self.max_n_sen: int = config.pop("max_n_sen")
        self.max_n_sen_add_val: int = config.pop("max_n_sen_add_val")
        self.max_t_len: int = config.pop("max_t_len")
        self.max_v_len: int = config.pop("max_v_len")
        self.type_vocab_size: int = config.pop("type_vocab_size")
        self.word_vec_size: int = config.pop("word_vec_size")

        # technical
        self.random_seed: Optional[int] = config.pop("random_seed")
        self.use_cuda: bool = config.pop("use_cuda")
        self.debug: bool = config.pop("debug")
        self.cudnn_enabled: bool = config.pop("cudnn_enabled")
        self.cudnn_benchmark: bool = config.pop("cudnn_benchmark")
        self.cudnn_deterministic: bool = config.pop("cudnn_deterministic")
        self.cuda_non_blocking: bool = config.pop("cuda_non_blocking")
        self.fp16_train: bool = config.pop("fp16_train")
        self.fp16_val: bool = config.pop("fp16_val")

        # model
        self.attention_probs_dropout_prob: float = config.pop(
            "attention_probs_dropout_prob"
        )
        self.hidden_dropout_prob: float = config.pop("hidden_dropout_prob")
        self.clip_dim: int = config.pop("clip_dim")
        self.hidden_size: int = config.pop("hidden_size")
        self.intermediate_size: int = config.pop("intermediate_size")
        self.layer_norm_eps: float = config.pop("layer_norm_eps")
        self.memory_dropout_prob: float = config.pop("memory_dropout_prob")
        self.num_attention_heads: int = config.pop("num_attention_heads")
        self.num_hidden_layers: int = config.pop("num_hidden_layers")
        self.n_memory_cells: int = config.pop("n_memory_cells")
        self.share_wd_cls_weight: bool = config.pop("share_wd_cls_weight")
        self.fix_emb: bool = config.pop("fix_emb")
        self.cross_attention_layers: int = config.pop("cross_attention_layers")
        self.ca_embedder: str = config.pop("ca_embedder")

        # uniter
        self.uniter_hidden_size: int = config.pop('uniter_hidden_size')
        self.uniter_hidden_dropout_prob: float = config.pop('uniter_hidden_dropout_prob')
        self.uniter_img_dim: int = config.pop('uniter_img_dim')
        self.uniter_pos_dim: int = config.pop("uniter_pos_dim")

        # optimization
        self.ema_decay: float = config.pop("ema_decay")
        self.initializer_range: float = config.pop("initializer_range")
        self.lr: float = config.pop("lr")
        self.lr_warmup_proportion: float = config.pop("lr_warmup_proportion")
        self.infty: int = config.pop("infty", 0)
        self.eps: float = config.pop("eps", 1e-6)

        # max position embeddings is calculated as the max joint sequence length
        self.max_position_embeddings: int = self.max_v_len + self.max_t_len

        # must be set manually as it depends on the dataset
        self.vocab_size: Optional[int] = None

        # knn
        self.dstore_size: int = config.pop("dstore_size")
        self.dstore_keys_path: str = config.pop('dstore_keys_path')
        self.dstore_vals_path: str = config.pop('dstore_vals_path')
        self.k_num: int = config.pop('k_num')
        self.alpha: int = config.pop('alpha')
        self.dstore_id_num: int = config.pop("dstore_id_num")
        self.knn_temperature: int = config.pop('knn_temperature')

        # assert the config is valid
        if self.share_wd_cls_weight:
            assert self.word_vec_size == self.hidden_size, (
                "hidden size has to be the same as word embedding size when "
                "sharing the word embedding weight and the final classifier weight"
            )

        # infer model type
        self.model_type = "re"

        self.post_init()

    def post_init(self):
        """
        Check config dict for correctness and raise

        Returns:
        """
        if self.strict:
            utils.check_config_dict(self.__class__.__name__, self.config)


class MetersConst(ConstantHolder):
    """
    Additional metric fields.
    """

    TRAIN_LOSS_PER_WORD = "train/loss_word"
    TRAIN_ACC = "train/acc"

    VAL_LOSS_PER_WORD = "val/loss_word"
    VAL_ACC = "val/acc"
    GRAD = "train/grad"


def update_config_from_args(config: Dict, args: argparse.Namespace, *, verbose: bool = True) -> Dict[str, Any]:
    """
    Modify config and paths given script arguments.

    Args:
        config: Config dictionary.
        args: Arguments.
        verbose: Print message when updating the config.

    Returns:
        Updated config dict.
    """
    # parse the --config inline modifier
    if args.config is not None:
        # get all fields to update from the argument and loop them
        update_fields: List[str] = args.config.split(",")
        for field_value in update_fields:
            # get field and value
            fields_str, value = field_value.strip().split("=")
            # convert value if necessary
            try:
                value = float(value)
                if round(value) == value:
                    value = int(value)
            except ValueError:
                pass
            if str(value).lower() == "true":
                value = True
            elif str(value).lower() == "false":
                value = False
            # update the correct nested dictionary field
            fields = fields_str.split(".")
            current_dict = config
            for i, field in enumerate(fields):
                if i == len(fields) - 1:
                    # update field
                    if field not in current_dict:
                        assert "same_as" in current_dict, (
                            f"Field {fields_str} not found in config {list(current_dict.keys())}. "
                            f"Typo or field missing in config.")
                    current_dict[field] = value
                    if verbose:
                        print(f"    Change config: Set {fields_str} = {value}")
                    break
                # go one nesting level deeper
                current_dict = current_dict[field]

    if args.workers is not None:
        config["dataset_train"]["num_workers"] = int(args.workers)
        config["dataset_val"]["num_workers"] = int(args.workers)
        if verbose:
            print(f"    Change config: Set dataloader workers to {args.workers} for train and val.")

    if args.seed is not None:
        if str(args.seed).lower() in ["none", "null"]:
            config["random_seed"] = None
        else:
            config["random_seed"] = int(args.seed)
        if verbose:
            print(f"    Change config: Set seed to {args.seed}. Deterministic")

    if args.no_cuda:
        config["use_cuda"] = False
        if verbose:
            print(f"    Change config: Set use_cuda to False.")
    
    # dataset / model
    if args.debug:
        config["debug"] = True
        if verbose:
            print(f"    Change config: Set debug to True")
    if args.dataset_max is not None:
        assert args.dataset_max > 0, "--dataset_max must be positive int."
        config["dataset_train"]["max_datapoints"] = args.dataset_max
        config["dataset_val"]["max_datapoints"] = args.dataset_max
        if verbose:
            print(
                f"    Change config: Set dataset_(train|val).max_datapoints to {args.dataset_max}"
            )
    config["train"]["batch_size"] = args.batch_size
    config["label_smoothing"] = args.label_smoothing

    return config

class TrainConfig(ConfigClass):
    """
    Base configuration class for training.

    Args:
        config: Configuration dictionary to be loaded, training part.
    """

    def __init__(self, config: Dict) -> None:
        self.batch_size: int = config.pop("batch_size")
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        self.num_epochs: int = config.pop("num_epochs")
        assert isinstance(self.num_epochs, int) and self.num_epochs > 0
        self.loss_func: str = config.pop("loss_func")
        assert isinstance(self.loss_func, str)
        self.clip_gradient: float = config.pop("clip_gradient")
        assert isinstance(self.clip_gradient, (int, float)) and self.clip_gradient >= -1


class ValConfig(ConfigClass):
    """
    Base configuration class for validation.

    Args:
        config: Configuration dictionary to be loaded, validation part.
    """

    def __init__(self, config: Dict) -> None:
        self.batch_size: int = config.pop("batch_size")
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        self.val_freq: int = config.pop("val_freq")
        assert isinstance(self.val_freq, int) and self.val_freq > 0
        self.val_start: int = config.pop("val_start")
        assert isinstance(self.val_start, int) and self.val_start >= 0
        self.det_best_field: str = config.pop("det_best_field")
        assert isinstance(self.det_best_field, str)
        self.det_best_compare_mode: str = config.pop("det_best_compare_mode")
        assert isinstance(self.det_best_compare_mode, str) and self.det_best_compare_mode in ["min", "max"]
        self.det_best_threshold_mode: str = config.pop("det_best_threshold_mode")
        assert isinstance(self.det_best_threshold_mode, str) and self.det_best_threshold_mode in ["rel", "abs"]
        self.det_best_threshold_value: float = config.pop("det_best_threshold_value")
        assert isinstance(self.det_best_threshold_value, (int, float)) and self.det_best_threshold_value >= 0
        self.det_best_terminate_after: float = config.pop("det_best_terminate_after")
        assert isinstance(self.det_best_terminate_after, int) and self.det_best_terminate_after >= -1


class SavingConfig(ConfigClass):
    """
    Base Saving Configuration Class

    Args:
        config: Configuration dictionary to be loaded, saving part.

    Attributes:
        keep_freq: Frequency to keep epochs. 1: Save after each epoch. Default -1: Keep nothing except best and last.
        save_last: Keep last epoch. Needed to continue training. Default: true
        save_best: Keep best epoch. Default: true
        save_opt_state: Save optimizer and lr scheduler. Needed to continue training. Default: true
    """

    def __init__(self, config: Dict) -> None:
        self.keep_freq: int = config.pop("keep_freq")
        self.save_last: bool = config.pop("save_last")
        self.save_best: bool = config.pop("save_best")
        self.save_opt_state: bool = config.pop("save_opt_state")
        assert self.keep_freq >= -1

class LoggingConfig(ConfigClass):
    """
    Base Logging Configuration Class

    Args:
        config: Configuration dictionary to be loaded, logging part.
    """

    def __init__(self, config: Dict) -> None:
        self.step_train: int = config.pop("step_train")
        self.step_val: int = config.pop("step_val")
        self.step_gpu: int = config.pop("step_gpu")
        self.step_gpu_once: int = config.pop("step_gpu_once")
        assert self.step_train >= -1
        assert self.step_val >= -1
        assert self.step_gpu >= -1
        assert self.step_gpu_once >= -1

class TrainerState(baseconfig.SaveableBaseModel):
    """
    Current trainer state that must be saved for training continuation..
    """
    # total time bookkeeping
    time_total: float = 0
    time_val: float = 0
    
    # state info TO SAVE
    start_epoch: int = 0
    current_epoch: int = 0
    epoch_step: int = 0
    total_step: int = 0
    det_best_field_current: float = 0
    det_best_field_best: Optional[float] = None

    # state info lists
    infos_val_epochs: List[int] = []
    infos_val_steps: List[int] = []
    infos_val_is_good: List[int] = []

    # logging
    last_grad_norm: int = 0



def get_config_file(args: argparse.Namespace) -> Tuple[str, str, str]:
    """
    Summary:
        使用するconfig fileを返す
        指定されていない場合 defaultのconfig fileを返す
    """
    if args.config_file is None:
        config_file = Path(args.config_dir, f"default.yaml")
    else:
        config_file = args.config_file
    print(f"config file: {config_file}")
    
    return config_file