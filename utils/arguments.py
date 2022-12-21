import argparse
import logging
from typing import Any, Dict, List


def set_parser():
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    # set log option
    parser = add_default_args(parser)
    # set exp setting
    parser = add_exp_setting(parser)
    # set exp running setting (trainer)
    parser = add_trainer_setting(parser)
    # set mart setting
    parser = add_model_setting(parser)
    # set others setting
    parser.add_argument('--dpath', type=str, default="data", help="instread of repo_config.py")
    parser.add_argument("--load_model", type=str, default=None, help="Load model from file.")
    parser.add_argument("--print_model", action="store_true", help=f"Print model")
    parser.add_argument('--datatype', type=str, default="bila", choices=['bila', 'bilas'])
    parser.add_argument('--wandb', '-w', action="store_true")
    parser.add_argument('--show_log', '-l', action="store_true")
    parser.add_argument('--del_weights', '-dw', action="store_true")

    # knn & test
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--make_knn_dstore', action="store_true")
    parser.add_argument('--do_knn', action="store_true")

    return parser.parse_args()


def add_default_args(parser: argparse.ArgumentParser) -> None:
    """
    Set logging options -q, -v and test flag -t
    """
    group = parser.add_mutually_exclusive_group()
    group.set_defaults(log_level=logging.INFO)
    group.add_argument(
        "-v", "--verbose", help="Verbose (debug) logging",
        action="store_const", const=logging.DEBUG, dest="log_level")
    group.add_argument(
        "-q", "--quiet", help="Silent mode, only log warnings",
        action="store_const", const=logging.WARN, dest="log_level")
    group.add_argument(
        "--log", help="Set log level manually", type=str, dest="log_level")
    
    return parser


def add_exp_setting(parser: argparse.ArgumentParser) -> None:
    """
    Set exp setting to parser
    """
    parser.add_argument("-c", "--config_file", type=str, default=None,
                        help="Specify either config file location or experiment group and name.")
    parser.add_argument("-d", "--data_dir", type=str, default="data", help="path to data")
    parser.add_argument("-n", "--num_runs", type=int, default=1, help="How many runs to do.")
    parser.add_argument("-a", "--start_run", type=int, default=1, help="Start at which run number.")
    parser.add_argument("-r", "--run_name", type=str, default="run",
                        help="Run name to save the model. Must not contain underscores.")
    
    return parser

def add_trainer_setting(parser: argparse.ArgumentParser):
    """
    Set exp running setting to parser
    """
    # configuration loading
    parser.add_argument("-o", "--config", type=str, default=None,
                        help="Modify the loaded YAML config. E.g. to change the number of dataloader workers "
                                "and the batchsize, use '-c dataloader.num_workers=20;train.batch_size=32'")
    parser.add_argument("--bs", "--batch_size", dest="batch_size", type=int, default=16, help="batch size")
    parser.add_argument("-ls", "--label_smoothing", dest="label_smoothing", type=float, default=0.1, help="label smoothing")
    parser.add_argument("--print_config", action="store_true", help="Print the experiment config.")
    
    # num workers
    parser.add_argument("--workers", type=int, default=None, help="Shortcut for setting dataloader workers.")
    
    # dataset path
    parser.add_argument("--config_dir", type=str, default="config", help="Folder with config files.")
    parser.add_argument("--log_dir", type=str, default="results")
    parser.add_argument("--data_path", type=str, default=None, help="Change the data path.")
    
    # checkpoint loading
    parser.add_argument("--load_epoch", type=int, default=None, help="Load epoch number.")
    parser.add_argument("--load_best", action="store_true", help="Load best epoch.")
    
    # validation
    parser.add_argument("--validate", action="store_true", help="Validation only.")
    parser.add_argument("--ignore_untrained", action="store_true", help="Validate even if no checkpoint was loaded.")
    
    # reset (delete everything)
    parser.add_argument("--reset", action="store_true", help="Delete experiment.")
    parser.add_argument("--seed", type=str, default=None,
                        help="Set seed. integer or none/null for auto-generated seed.")
    
    # GPU
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA.")
    parser.add_argument("--single_gpu", action="store_true", help="Disable multi GPU with nn.DataParallel.")

    return parser

def add_model_setting(parser: argparse.ArgumentParser) -> None:
    """
    Add some additional arguments that are required for mart.

    Args:
        parser: Command line argument parser.
    """
    # paths
    parser.add_argument(
        "--coot_feat_dir",
        type=str,
        default="provided_embeddings",
        help="COOT Embeddings dir.",
    )
    parser.add_argument(
        "--video_feature_dir",
        type=str,
        default="data/mart_video_feature",
        help="Dir containing the video features",
    )

    # Technical
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_preload", action="store_true")
    parser.add_argument(
        "--dataset_max", type=int, default=None, help="Reduce dataset size for testing."
    )

    return parser

def add_mart_args(parser: argparse.ArgumentParser) -> None:
    """
    Add some additional arguments that are required for mart.

    Args:
        parser: Command line argument parser.
    """
    # paths
    parser.add_argument(
        "--coot_feat_dir",
        type=str,
        default="provided_embeddings",
        help="COOT Embeddings dir.",
    )
    parser.add_argument(
        "--annotations_dir", type=str, default="annotations", help="Annotations dir."
    )
    parser.add_argument(
        "--video_feature_dir",
        type=str,
        default="data/mart_video_feature",
        help="Dir containing the video features",
    )

    # Technical
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_preload", action="store_true")
    parser.add_argument(
        "--dataset_max", type=int, default=None, help="Reduce dataset size for testing."
    )

    return parser


def parse_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Summary:
        argparseとyamlファイルのconfigを共存させる
    Returns:
        Namespace: argparserと同様に使用可能
    Note:
        configより引数に指定した値が優先
    """
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {
            arg[2:].split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")
        }
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    return args