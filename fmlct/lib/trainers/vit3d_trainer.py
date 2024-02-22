import math
import os
import time
import torch
from torchmetrics import AUROC, AveragePrecision
import timm.optim

from fmlct.lib.utils.mod_from_moco import AverageMeter, ProgressMeter

from .base_trainer import BaseTrainer

__all__ = ['ViT3DTrainer']

class ViT3DTrainer(BaseTrainer):
    r"""
    ViT 3D Trainer
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
            linear_keyword = 'head'
            
            # load pretrained weights
            if args.pretrain is not None and os.path.exists(args.pretrain):
                print(f"=> Start loading pretrained weights from {args.pretrain}")
                checkpoint = torch.load(args.pretrain, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                for k in list(state_dict.keys()):
                    # retain only base_encoder up to before the embedding layer
                    if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                        # remove prefix
                        state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                    if k.startswith('base_encoder') and not k.startswith('base_encoder.%s' % linear_keyword):
                        # remove prefix
                        state_dict[k[len("base_encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k] 
                msg = self.model.load_state_dict(state_dict, strict=False)
                # import pickle
                # pickle.dump(self.model, open("/mnt/tmp/model_new.pkl", "wb"))
                # with open("/mnt/tmp/model_new.txt", "w") as f:
                #     f.write(str(self.model))
                print(f'Loading messages: \n {msg}')
                print(f"=> Finish loading pretrained weights from {args.pretrain}")
            
            # self.model = torch.nn.Sequential(self.model, torch.nn.Linear(1000, args.num_classes))
            # freeze all layers but the last fc
            if args.get("freeze_all_except_fc", False):
                for name, param in self.model.named_parameters():
                    if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
                        param.requires_grad = False
                # init the fc layer
                getattr(self.model, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
                getattr(self.model, linear_keyword).bias.data.zero_() 
                
            self.loss_fn = torch.nn.CrossEntropyLoss().cuda(args.gpu)

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

        print("=> changing learning rate to match batch size, lr *= batch_size / 256: ", args.lr)
        args.lr = args.lr * args.batch_size / 256

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
            val_dataset = args.val_dataset_obj

            print(f"=> Train dataset length: {len(train_dataset)}")
            print(f"=> Val dataset length: {len(val_dataset)}")

            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            else:
                train_sampler = None
                val_sampler = None

            self.dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                          batch_size=self.batch_size, 
                                                          shuffle=(train_sampler is None),
                                                          num_workers=self.workers, 
                                                          pin_memory=True, 
                                                          sampler=train_sampler, 
                                                          drop_last=True)
            self.iters_per_epoch = len(self.dataloader)

            self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                              batch_size=self.batch_size, 
                                                              shuffle=(val_sampler is None),
                                                              num_workers=self.workers, 
                                                              pin_memory=True, 
                                                              sampler=val_sampler, 
                                                              drop_last=False)
            self.val_iters_per_epoch = len(self.val_dataloader)

        else:
            raise ValueError(f"Dataloader has been created. Do not create twice.")
        
    def run(self):
        args = self.args
        # Compute iterations when resuming
        niters = args.start_epoch * self.iters_per_epoch

        best_metric = 0
        for epoch in range(args.start_epoch, args.epochs):
            print(args.ckpt_dir)
            if args.distributed:
                self.dataloader.sampler.set_epoch(epoch)
                torch.distributed.barrier()

            # evaluate on validation set
            self.evaluate(epoch, niters)

            # train for one epoch
            this_metric = self.epoch_train(epoch, niters)
            # save checkpoint
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
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

                    if getattr(args, "save_best", False) and this_metric > best_metric:
                        best_metric = this_metric
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
        losses = AverageMeter('Loss', ':.4e')
        auc = AverageMeter('AUC', ':6.2f')
        ap = AverageMeter('AP', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, auc, ap],
            prefix="Epoch: [{}]".format(epoch))

        """
        Switch to eval mode:
        Under the protocol of linear classification on frozen features/models,
        it is not legitimate to change any part of the pre-trained model.
        BatchNorm in train mode may revise running mean/std (even if it receives
        no gradient), which are part of the model parameters too.
        """
        model.eval()

        end = time.time()
        iters_per_epoch = len(train_loader)
        for i, (image, target) in enumerate(train_loader):
            if image.isnan().any():
                print("images nan detected")
                continue

            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                image = image.to(args.gpu, non_blocking=True)
                target = target.to(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(image)
                loss = loss_fn(output, target)
                if loss.isnan().any():
                    print("loss nan detected")

            # compute gradient and do SGD step
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = image.size(0)
            losses.update(loss.item(), batch_size)
            this_auc = AUROC(task="multiclass", num_classes=args.num_classes)(output, target)
            this_ap = AveragePrecision(task="multiclass", num_classes=args.num_classes)(output, target)
            auc.update(this_auc, batch_size)
            ap.update(this_ap, batch_size)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.print_freq > 0 and i % args.print_freq == 0:
                progress.display(i)
            
            if args.rank == 0:
                args.summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)
                args.summary_writer.add_scalar("auc", this_auc, epoch * iters_per_epoch + i)
                args.summary_writer.add_scalar("ap", this_ap, epoch * iters_per_epoch + i)
              
        print(' @ Train: Loss {losses.avg:.5f} AUC {auc.avg:.3f} AP {ap.avg:.3f}'
            .format(losses=losses, auc=auc, ap=ap))
        return this_auc


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
            msg = self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            print(f'=> Loading messages: \n {msg}')
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        
    @torch.no_grad()
    def evaluate(self, epoch=0, niters=0):
        args = self.args
        model = self.wrapped_model
        val_loader = self.val_dataloader
        loss_fn = self.loss_fn

        # meters
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        auc = AverageMeter('AUC', ':6.2f')
        ap = AverageMeter('AP', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, auc, ap],
            prefix='Test: ')

        # switch to evaluate mode
        model.eval()

        end = time.time()
        iters_per_epoch = len(val_loader)
        for i, (image, target) in enumerate(val_loader):
            if args.gpu is not None:
                image = image.to(args.gpu, non_blocking=True)
                target = target.to(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(image)

                loss = loss_fn(output, target)

            batch_size = image.size(0)
            losses.update(loss.item(), batch_size)
            this_auc = AUROC(task="multiclass", num_classes=args.num_classes)(output, target)
            this_ap = AveragePrecision(task="multiclass", num_classes=args.num_classes)(output, target)
            auc.update(this_auc, batch_size)
            ap.update(this_ap, batch_size)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.val_print_freq > 0 and i % args.val_print_freq == 0:
                progress.display(i)
            
            if args.rank == 0:
                args.summary_writer.add_scalar("val_loss", loss.item(), epoch * iters_per_epoch + i)
                args.summary_writer.add_scalar("val_auc", this_auc, epoch * iters_per_epoch + i)
                args.summary_writer.add_scalar("val_ap", this_ap, epoch * iters_per_epoch + i)

        print(' * Test: AUC {auc.avg:.3f} AP {ap.avg:.3f}'
            .format(auc=auc, ap=ap))


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