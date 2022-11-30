import os
import shutil
import datetime
import numpy as np

from trainer import MartTrainer
from models.model import create_model
from datasets.bila import create_datasets_and_loaders
from utils.utils_yaml import load_yaml_config_file
from utils.utils import fix_seed
from utils.arguments import update_mart_config_from_args, set_parser, update_config_from_args
from utils.setting import setup_experiment_identifier_from_args
from utils.configs import MartConfig as Config


def main():
    args = set_parser()

    # load repository config yaml file to dict
    exp_group, exp_name, exp_type, config_file = setup_experiment_identifier_from_args(args)
    config = load_yaml_config_file(config_file)

    # update experiment config given the script arguments
    config = update_config_from_args(config, args)
    config = update_mart_config_from_args(config, args)

    # read experiment config dict
    cfg = Config(config)
    if args.print_config:
        print(cfg)

    # set seed
    verb = "Set seed"
    if cfg.random_seed is None:
        cfg.random_seed = np.random.randint(0, 2 ** 15, dtype=np.int32)
        verb = "Randomly generated seed"
    print(f"{verb} {cfg.random_seed} deterministic {cfg.cudnn_deterministic} "
            f"benchmark {cfg.cudnn_benchmark}")
    fix_seed(cfg.random_seed, cudnn_deterministic=cfg.cudnn_deterministic, cudnn_benchmark=cfg.cudnn_benchmark)

    # create dataset
    train_set, train_loader, _, val_loader, _, test_loader =\
        create_datasets_and_loaders(cfg, args.data_dir, datatype=args.datatype)

    for i in range(args.start_run):
        run_number = datetime.datetime.now()
        run_name = f"{args.run_name}{run_number}"

        model = create_model(cfg, len(train_set.word2idx), cache_dir=args.cache_dir)

        if args.print_model and i == 0:
            print(model)

        # always load best epoch during validation
        load_best = args.load_best or args.validate

        trainer = MartTrainer(
            cfg, model, exp_group, exp_name, exp_type, run_name, len(train_loader), log_dir=args.log_dir,
            log_level=args.log_level, logger=None, print_graph=args.print_graph, reset=args.reset, load_best=load_best,
            load_epoch=args.load_epoch, load_model=args.load_model, inference_only=args.validate,
            annotations_dir=args.data_dir)

        if args.validate:
            if not trainer.load and not args.ignore_untrained:
                raise ValueError("Validating an untrained model! No checkpoints were loaded. Add --ignore_untrained "
                                    "to ignore this error.")
            trainer.validate_epoch(val_loader, datatype=args.datatype)
        else:
            trainer.train_model(
                    train_loader, val_loader, 
                    test_loader, datatype=args.datatype,
                    use_wandb=args.wandb, show_log=args.show_log,)
        
        if args.del_weights:
            print('Pay Attention : Delete All Model weights ... ', end='')
            weights_dir = os.path.join(trainer.exp.path_base, "models")
            shutil.rmtree(weights_dir)
            print('ok')

        trainer.close()
        del model
        del trainer


if __name__ == "__main__":
    main()