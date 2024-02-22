import math
import os
import time
import torch

from fmlct.lib.utils.mod_from_moco import AverageMeter, ProgressMeter

from .base_trainer import BaseTrainer
import timm.optim

__all__ = ['MoCo3DTrainer']

class MoCo3DTrainer(BaseTrainer):
    r"""
    MoCo V3 3D Trainer
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = args.model_name
        self.scaler = torch.cuda.amp.GradScaler()

    def build_model(self):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args

            print(f"=> creating model {self.model_name} of arch {args.arch}")
            self.model = args.model_obj
            
            # load pretrained weights
            if args.pretrain is not None and os.path.exists(args.pretrain):
                print(f"=> Start loading pretrained weights from {args.pretrain}")
                checkpoint = torch.load(args.pretrain, map_location='cpu')
                raise ValueError("=> Pretrain is not supported yet")
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                if args.pretrain_load == 'enc+dec':
                    msg = self.model.load_state_dict(state_dict, strict=False)
                elif args.pretrain_load == 'enc':
                    state_dict = {k[len("encoder."):]:v for k,v in state_dict.items() if k.startswith('encoder.')}
                    msg = self.model.encoder.load_state_dict(state_dict, strict=False)
                elif args.pretrain_load == 'dec':
                    state_dict = {k[len("decoder."):]:v for k,v in state_dict.items() if k.startswith('decoder.')}
                    msg = self.model.decoder.load_state_dict(state_dict, strict=False)
                else:
                    raise ValueError(f"=> Wrong pretrain_load: {args.pretrain_load}")
                self.model.encoder.head.weight.data.normal_(mean=0.0, std=0.01)
                self.model.encoder.head.bias.data.zero_()
                print(f'Loading messages: \n {msg}')
                print(f"=> Finish loading pretrained weights from {args.pretrain}")

            self.loss_fn = torch.nn.Identity()

            self.wrap_model()
        elif self.model_name == 'Unknown':
            raise ValueError("=> Model name is still unknown")
        else:
            raise ValueError("=> Model has been created. Do not create twice")

    def build_optimizer(self):
        assert(self.model is not None and self.wrapped_model is not None), \
                "Model is not created and wrapped yet. Please create model first."
        print("=> creating optimizer")
        args = self.args

        optim_params = self.get_parameter_groups()

        print(f"Changing learning rate to match batch size, lr *= batch_size / {args.get('pretrain_batch_size', 256)}: ", args.lr)
        args.lr = args.lr * args.batch_size / args.get("pretrain_batch_size", 256)

        if args.optimizer == 'lars':
            self.optimizer = timm.optim.LARS(optim_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(optim_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError(f"Unsupported optimizer {args.optimizer}")
        
    def build_dataloader(self):
        if self.dataloader is None:
            print("=> creating train dataloader")
            args = self.args

            train_dataset = args.train_dataset_obj

            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            else:
                train_sampler = None

            self.dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                          batch_size=self.batch_size, 
                                                          shuffle=(train_sampler is None),
                                                          num_workers=self.workers, 
                                                          pin_memory=True, 
                                                          sampler=train_sampler, 
                                                          drop_last=True)
            self.iters_per_epoch = len(self.dataloader)
        else:
            raise ValueError(f"Dataloader has been created. Do not create twice.")
        
    def run(self):
        args = self.args
        # Compute iterations when resuming
        niters = args.start_epoch * self.iters_per_epoch

        best_metric = torch.inf
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                self.dataloader.sampler.set_epoch(epoch)
                torch.distributed.barrier()

            # train for one epoch
            loss = self.epoch_train(epoch, niters)

            # save checkpoint
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                if (epoch + 1) % args.save_freq == 0:
                    self.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': self.model.state_dict(),
                            'optimizer' : self.optimizer.state_dict(),
                            'scaler': self.scaler.state_dict(), # additional line compared with base imple
                        }, 
                        is_best=False, 
                        filename=f'{args.ckpt_dir}/checkpoint_{epoch:04d}.pth.tar'
                    )
                    if getattr(args, "save_best", False) and loss < best_metric:
                        best_metric = loss
                        self.save_checkpoint(
                            {
                                'epoch': epoch + 1,
                                'arch': args.arch,
                                'state_dict': self.model.state_dict(),
                                'optimizer' : self.optimizer.state_dict(),
                                'scaler': self.scaler.state_dict(), # additional line compared with base imple
                            }, 
                            is_best=False,
                            filename=f'{args.ckpt_dir}/best.pth.tar'
                        )

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        model = self.wrapped_model
        optimizer = self.optimizer
        scaler = self.scaler
        loss_fn = self.loss_fn

        # meters
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        learning_rates = AverageMeter('LR', ':.4e')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, learning_rates, losses],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()

        end = time.time()
        iters_per_epoch = len(train_loader)
        moco_m = args.moco_m
        for i, (images, _) in enumerate(train_loader):
            if images[0].isnan().any() or images[1].isnan().any():
                print("images nan detected")
                continue

            # measure data loading time
            data_time.update(time.time() - end)

            # adjust learning rate and momentum coefficient per iteration
            lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
            learning_rates.update(lr)
            if args.moco_m_cos:
                moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

            (images[0].shape, images[1].shape)
            if args.gpu is not None:
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(True):
                loss = model(images[0], images[1], moco_m)
                if loss.isnan().any():
                    print("loss nan detected")

            losses.update(loss.item(), images[0].size(0))
            if args.rank == 0:
                args.summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)
                args.summary_writer.add_scalar("epoch", epoch, epoch * iters_per_epoch + i)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                if args.rank == 0:
                    args.summary_writer.add_text('info', progress.get_display(i), epoch * iters_per_epoch + i)

    def resume(self):
        args = self.args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        

def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m