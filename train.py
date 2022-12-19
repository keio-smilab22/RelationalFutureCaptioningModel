import os
import shutil
import datetime
import numpy as np

from trainer import Trainer
from models.model import create_model
from datasets.bila import create_datasets_and_loaders
from utils.utils import fix_seed, load_yaml_to_config
from utils.arguments import set_parser
from utils.configs import Config, get_config_file, update_config_from_args


def main():
    args = set_parser()

    # load config file to dict
    config_file = get_config_file(args)
    config = load_yaml_to_config(config_file)

    # update config given the arguments
    config = update_config_from_args(config, args)

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
        run_date = datetime.datetime.now()
        run_name = f"{args.run_name}{run_date}"

        model = create_model(cfg, len(train_set.word2idx))

        if args.print_model and i == 0:
            print(model)

        # always load best epoch during validation
        load_best = args.load_best or args.validate # False

        trainer = Trainer(
            cfg, model, run_name, len(train_loader), log_dir=args.log_dir,
            log_level=args.log_level, logger=None, reset=args.reset, load_best=load_best,
            load_epoch=args.load_epoch, load_model=args.load_model, is_test=args.validate,
            data_dir=args.data_dir, show_log=args.show_log,)

        if args.validate:
            if not trainer.load and not args.ignore_untrained:
                raise ValueError("Validating an untrained model! No checkpoints were loaded. Add --ignore_untrained "
                                    "to ignore this error.")
            trainer.validate_epoch(val_loader, datatype=args.datatype)
        else:
            trainer.train_model(
                    train_loader, val_loader, 
                    test_loader, datatype=args.datatype,
                    use_wandb=args.wandb, show_log=args.show_log,
                    make_knn_dstore=args.make_knn_dstore,
                    do_knn=args.do_knn)
        
        if args.del_weights:
            print('Pay Attention : Delete All Model weights ... ', end='')
            weights_dir = os.path.join(trainer.exp.path_base, "models")
            shutil.rmtree(weights_dir)
            print('ok')

        if args.make_knn_dstore:
            print(f'Save dstore Num >>>> {trainer.translator.dstore_idx}')

        trainer.close()
        del model
        del trainer


if __name__ == "__main__":
    main()
