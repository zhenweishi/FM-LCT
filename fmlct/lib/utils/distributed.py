import torch
import torch.distributed as dist
import builtins

__all__ = [
    "concat_all_gather",
    "dist_setup",
]

def dist_setup(ngpus_per_node, args):
    torch.multiprocessing.set_start_method('fork', force=True)
    # suppress printing if not master
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        dist.barrier()

@torch.no_grad()
def concat_all_gather(tensor, distributed=True):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if distributed:
        dist.barrier()
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(dist.get_world_size())]
        # print(f"World size: {dist.get_world_size()}")
        dist.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        return tensor