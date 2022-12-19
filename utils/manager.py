import shutil
import os
import time
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any

import numpy as np
import torch
from torch import nn

from utils.utils import TrainerPathConst
from utils.configs import Config, TrainerState

class ModelManager:
    """
    Wrapper for MART models.
    """

    def __init__(self, cfg: Config, model: nn.Module):
        # update config type hints
        self.model_dict: Dict[str, nn.Module] = {"model": model}
        self.was_loaded: bool = False
        self.cfg: Config = cfg
        self.is_train = True
    
    def is_autocast_enabled(self) -> bool:
        """
        Given train or val state and config, determine whether autocast should be enabled.

        Returns:
            Bool.
        """
        return self.cfg.fp16_train if self.is_train else self.cfg.fp16_val

    def get_all_params(self) -> Tuple[Any, Any, Any]:
        """
        Since there are multiple networks used by this trainer, this
        function can be used to get all the parameters at once.


        Returns:
            params, param_names, params_flat
        """
        # loop models and collect parameters
        params, param_names, params_flat = [], [], []
        for _model_name, model in self.model_dict.items():
            _params, _param_names, _params_flat = self.get_params_opt_simple(model)
            params.extend(_params)
            param_names.extend(_param_names)
            params_flat.extend(_params_flat)
        return params, param_names, params_flat

    def set_all_models_train(self) -> None:
        """
        Set all networks to train mode.
        """
        self.is_train = True
        for model in self.model_dict.values():
            model.train()

    def set_all_models_eval(self) -> None:
        """
        Set all networks to eval mode.
        """
        self.is_train = False
        for model in self.model_dict.values():
            model.eval()

    def get_model_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get all state dicts of all networks into a single variable

        Returns:
            Dict with model names and keys and state dict of the model as value.
        """
        return_dict = {}
        for model_name, model in self.model_dict.items():
            return_dict[model_name] = model.state_dict()
        return return_dict

    def set_model_state(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """
        Use the dict of state dicts created by get_model_state to load all network weights.

        Args:
            state: Dict with model names and keys and state dict of the model as value.
        """
        self.was_loaded = True

        # backwards compatibility to coot-videotext
        if isinstance(state, list):
            for i, model_name in enumerate(self.model_dict.keys()):
                print(f"Backward compatible loading for coot-videotext: {model_name}")
                this_state = state[i]
                new_state = {}
                for param_name, param in this_state.items():
                    for replace_from, replace_to in {
                        "input_norm.": "norm_input.",
                        "input_fc.": "input_fc.mlp.",
                        "pooler.genpool": "pooler.pools.0.genpool"
                    }.items():
                        param_name = param_name.replace(replace_from, replace_to)
                    new_state[param_name] = param
                self.model_dict[model_name].load_state_dict(new_state)
            return
        # backwards compatibility to recurrent_transformer (original MART repository style checkpoints)
        if sorted(list(state.keys())) == ["epoch", "model", "model_cfg", "opt"]:
            state_dict = state["model"]
            print(f"Backward compatible loading for recurrent_transformer epoch {state['epoch']} with "
                  f"{sum([np.product(param.shape) for param in state_dict.values()])} parameters")
            self.model_dict['model'].load_state_dict(state_dict)
            return
        # newest version of loading. keys in the state correspond to keys in the model_dict.
        for model_name, state_dict in state.items():
            self.model_dict[model_name].load_state_dict(state_dict)

    def get_params_opt_simple(self, model: nn.Module) -> (
            Tuple[List[Dict[str, Any]], List[str], List[torch.Tensor]]):
        """
        Args:
            model: Model to get the parameters from.

        Returns:
            Tuple of:
                List of:
                    Dict of:
                        'params': The parameter
                        'decay_mult': Multiply weight decay with this factor
                        'lr_mult': Multiplay learning rate with this factor
                List of:
                    parameter names
                List of:
                    parameters
        """
        params_dict: Dict[str, torch.Tensor] = dict(model.named_parameters())
        params, param_names, params_flat = [], [], []
        # print(cfg.training.representation)
        for key, value in params_dict.items():
            decay_mult = 1.0
            if self.cfg.optimizer.weight_decay_for_bias and 'bias' in key:
                decay_mult = 0.0
            params += [{
                'params': value,
                'decay_mult': decay_mult,
                'lr_mult': 1.0
            }]
            param_names += [key]
            params_flat += [value]

        return params, param_names, params_flat


class FilesHandler:
    """
    Overwrite default filehandler to add some more paths.
    """

    def __init__(
        self,
        run_name: str,
        log_dir: str,
        data_dir: str,
    ):
        self.run_name: str = run_name
        self.data_dir = data_dir
    
        self.path_base: Path = Path(log_dir, "{}".format(self.run_name))
        self.path_caption = self.path_base / TrainerPathConst.DIR_CAPTION
        self.path_logs = self.path_base / TrainerPathConst.DIR_LOGS
        self.path_models = self.path_base / TrainerPathConst.DIR_MODELS
        self.path_metrics = self.path_base / TrainerPathConst.DIR_METRICS
        self.path_tensorb = self.path_base / TrainerPathConst.DIR_TB
        self.path_embeddings = self.path_base / TrainerPathConst.DIR_EMBEDDINGS


    def get_translation_files(self, epoch: Union[int, str], split: str, make_knn_dstore: bool=False) -> Path:
        """
        Summary:
            生成文・gt文の保存先のpathを返す
        Args:
            epoch: Epoch.
            split: dataset split (val, test)
        """
        if make_knn_dstore:
            return (
                self.path_caption
                / f"{TrainerPathConst.FILE_PREFIX_TRANSL_RAW}_{split}.json"
            )
        else:
            return (
                self.path_caption
                / f"{TrainerPathConst.FILE_PREFIX_TRANSL_RAW}_{epoch}_{split}.json"
            )

    def setup_dirs(self, reset: bool = False) -> None:
        """
        Summary:
            各種保存用のフォルダを作成
            resetの場合は存在するフォルダを削除後、新たに作成する

        Args:
            reset: Delete this experiment
        """
        if reset:
            # delete base path
            shutil.rmtree(self.path_base, ignore_errors=True)
            time.sleep(0.1)  # this avoids "cannot create dir that exists" on windows

        # create all paths
        for path in self.path_logs, self.path_models, self.path_metrics, self.path_tensorb:
            os.makedirs(path, exist_ok=True)
        os.makedirs(self.path_caption, exist_ok=True)

    def get_existing_checkpoints(self) -> List[int]:
        """
        Get list of all existing checkpoint numbers..

        Returns:
            List of checkpoint numbers.
        """
        # get list of existing trainerstate filenames
        list_of_files = glob.glob(str(self.get_trainerstate_file("*")))

        # extract epoch numbers from those filenames
        ep_nums = sorted([int(a.split(f"{TrainerPathConst.FILE_PREFIX_TRAINERSTATE}_")[-1].split(".json")[0])
                            for a in list_of_files])
        return ep_nums

    def find_best_epoch(self):
        """
        Find best episode out of existing checkpoint data.

        Returns:
            Best epoch or -1 if no epochs are found.
        """
        ep_nums = self.get_existing_checkpoints()
        if len(ep_nums) == 0:
            # no checkpoints found
            return -1

        # read trainerstate of the last epoch (contains all info needed to find the best epoch)
        temp_state = TrainerState.create_from_file(self.get_trainerstate_file(ep_nums[-1]))
        if len(temp_state.infos_val_epochs) == 0:
            # no validation has been done, assume last epoch is best
            return ep_nums[-1]

        # read the flags for each epoch that state whether that was a good or bad epoch
        # the last good epoch is the best one
        where_res = np.where(temp_state.infos_val_is_good)[0]
        best_idx = where_res[-1]
        best_epoch = temp_state.infos_val_epochs[best_idx]
        return best_epoch

    # ---------- File definitions. ----------
    # Parameter epoch allows str to create glob filenames with "*".

    def get_models_file(self, epoch: Union[int, str]) -> Path:
        """
        Get file path for storing the model.

        Args:
            epoch: Epoch.

        Returns:
            Path
        """
        return self.path_models / f"{TrainerPathConst.FILE_PREFIX_MODEL}_{epoch}.pth"

    def get_models_file_ema(self, epoch: Union[int, str]) -> Path:
        """
        Get file path for storing the model EMA weights.

        Args:
            epoch: Epoch.

        Returns:
            Path
        """
        return self.path_models / f"{TrainerPathConst.FILE_PREFIX_MODELEMA}_{epoch}.pth"

    def get_optimizer_file(self, epoch: Union[int, str]) -> Path:
        """
        Get file path for storing the model.

        Args:
            epoch: Epoch.

        Returns:
            Path
        """
        return self.path_models / f"{TrainerPathConst.FILE_PREFIX_OPTIMIZER}_{epoch}.pth"

    def get_trainerstate_file(self, epoch: Union[int, str]) -> Path:
        """
        Get file path for storing the state of the trainer. This is needed for currectly resuming training.

        Args:
            epoch: Epoch.

        Returns:
            Path
        """
        return self.path_models / f"{TrainerPathConst.FILE_PREFIX_TRAINERSTATE}_{epoch}.json"

    def get_metrics_step_file(self, epoch: Union[int, str]) -> Path:
        """
        Get file path for storing step-based metrics.

        Args:
            epoch: Epoch.

        Returns:
            Path
        """
        return self.path_metrics / f"{TrainerPathConst.FILE_PREFIX_METRICS_STEP}_{epoch}.json"

    def get_metrics_epoch_file(self, epoch: Union[int, str]) -> Path:
        """
        Get file path for storing epoch-based metrics.

        Args:
            epoch: Epoch.

        Returns:
            Path
        """
        return self.path_metrics / f"{TrainerPathConst.FILE_PREFIX_METRICS_EPOCH}_{epoch}.json"

class ModelManager:
    """
    Wrapper for models.
    """

    def __init__(self, cfg: Config, model: nn.Module):
        # update config type hints
        self.model_dict: Dict[str, nn.Module] = {"model": model}
        self.was_loaded: bool = False
        self.cfg: Config = cfg
        self.is_train = True
    
    def is_autocast_enabled(self) -> bool:
        """
        Given train or val state and config, determine whether autocast should be enabled.

        Returns:
            Bool.
        """
        return self.cfg.fp16_train if self.is_train else self.cfg.fp16_val

    def get_all_params(self) -> Tuple[Any, Any, Any]:
        """
        Since there are multiple networks used by this trainer, this
        function can be used to get all the parameters at once.


        Returns:
            params, param_names, params_flat
        """
        # loop models and collect parameters
        params, param_names, params_flat = [], [], []
        for _model_name, model in self.model_dict.items():
            _params, _param_names, _params_flat = self.get_params_opt_simple(model)
            params.extend(_params)
            param_names.extend(_param_names)
            params_flat.extend(_params_flat)
        return params, param_names, params_flat

    def set_all_models_train(self) -> None:
        """
        Set all networks to train mode.
        """
        self.is_train = True
        for model in self.model_dict.values():
            model.train()

    def set_all_models_eval(self) -> None:
        """
        Set all networks to eval mode.
        """
        self.is_train = False
        for model in self.model_dict.values():
            model.eval()

    def get_model_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get all state dicts of all networks into a single variable

        Returns:
            Dict with model names and keys and state dict of the model as value.
        """
        return_dict = {}
        for model_name, model in self.model_dict.items():
            return_dict[model_name] = model.state_dict()
        return return_dict

    def set_model_state(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """
        Use the dict of state dicts created by get_model_state to load all network weights.

        Args:
            state: Dict with model names and keys and state dict of the model as value.
        """
        self.was_loaded = True

        # backwards compatibility to coot-videotext
        if isinstance(state, list):
            for i, model_name in enumerate(self.model_dict.keys()):
                print(f"Backward compatible loading for coot-videotext: {model_name}")
                this_state = state[i]
                new_state = {}
                for param_name, param in this_state.items():
                    for replace_from, replace_to in {
                        "input_norm.": "norm_input.",
                        "input_fc.": "input_fc.mlp.",
                        "pooler.genpool": "pooler.pools.0.genpool"
                    }.items():
                        param_name = param_name.replace(replace_from, replace_to)
                    new_state[param_name] = param
                self.model_dict[model_name].load_state_dict(new_state)
            return
        # backwards compatibility to recurrent_transformer (original MART repository style checkpoints)
        if sorted(list(state.keys())) == ["epoch", "model", "model_cfg", "opt"]:
            state_dict = state["model"]
            print(f"Backward compatible loading for recurrent_transformer epoch {state['epoch']} with "
                  f"{sum([np.product(param.shape) for param in state_dict.values()])} parameters")
            self.model_dict['model'].load_state_dict(state_dict)
            return
        # newest version of loading. keys in the state correspond to keys in the model_dict.
        for model_name, state_dict in state.items():
            self.model_dict[model_name].load_state_dict(state_dict)

    def get_params_opt_simple(self, model: nn.Module) -> (
            Tuple[List[Dict[str, Any]], List[str], List[torch.Tensor]]):
        """
        Args:
            model: Model to get the parameters from.

        Returns:
            Tuple of:
                List of:
                    Dict of:
                        'params': The parameter
                        'decay_mult': Multiply weight decay with this factor
                        'lr_mult': Multiplay learning rate with this factor
                List of:
                    parameter names
                List of:
                    parameters
        """
        params_dict: Dict[str, torch.Tensor] = dict(model.named_parameters())
        params, param_names, params_flat = [], [], []
        # print(cfg.training.representation)
        for key, value in params_dict.items():
            decay_mult = 1.0
            if self.cfg.optimizer.weight_decay_for_bias and 'bias' in key:
                decay_mult = 0.0
            params += [{
                'params': value,
                'decay_mult': decay_mult,
                'lr_mult': 1.0
            }]
            param_names += [key]
            params_flat += [value]

        return params, param_names, params_flat