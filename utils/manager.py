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
