import os
import warnings

import torch
import torch.multiprocessing as mp
# mp.set_sharing_strategy('file_system')
import wandb
from torch.utils.tensorboard import SummaryWriter

import sys
# current_path = os.getcwd()
# delete_part = current_path.split('/')[-1]
# current_path = current_path.replace('/'+delete_part, '')
# sys.path.append(current_path)
import argparse

from fmlct.lib.utils import set_seed, dist_setup, get_conf
import fmlct.lib.trainers as trainers
from easydict import EasyDict as edict


def finetuning_main(path):
    args = get_conf(path, conf_parser='monai')
    # if "imports" in args:
    #     del args["imports"]
    # set seed if required
    args = edict(**args)
    if "imports" in args:
        del args["imports"]
    set_seed(args.get("seed", None))


    if not args.multiprocessing_distributed and args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.get("ngpus_per_node", None) is None:
        ngpus_per_node = torch.cuda.device_count()
        args.ngpus_per_node = ngpus_per_node
    else:
        ngpus_per_node = args.ngpus_per_node
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, 
                nprocs=ngpus_per_node, 
                args=(args,))
    else:
        print("single process")
        main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    ngpus_per_node = args.ngpus_per_node
    dist_setup(ngpus_per_node, args)

    # init trainer
    trainer_class = getattr(trainers, f'{args.trainer_name}', None)
    assert trainer_class is not None, f"Trainer class {args.trainer_name} is not defined"
    trainer = trainer_class(args)

    if args.rank == 0:
        if getattr(args, 'use_tensorboard', False):
            args.summary_writer = SummaryWriter(log_dir=args.log_dir)
        if getattr(args, 'use_wandb', False):        
            if args.wandb_id is None:
                args.wandb_id = wandb.util.generate_id()
                run = wandb.init(project=f"{args.proj_name}_{args.dataset}", 
                                name=args.run_name, 
                                config=vars(args),
                                id=args.wandb_id,
                                resume='allow',
                                dir=args.log_dir)

    # create model
    trainer.build_model()
    # create optimizer
    trainer.build_optimizer()
    # resume training
    if args.resume:
        trainer.resume()
    trainer.build_dataloader()

    trainer.run()

    if args.rank == 0:
        if getattr(args, 'use_tensorboard', False):
            args.summary_writer.close()
        if getattr(args, 'use_wandb', False):
            run.finish()

