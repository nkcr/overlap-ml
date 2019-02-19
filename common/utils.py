import os
import shutil
import torch
from torch.autograd import Variable

from datetime import datetime
from tensorboardX.summary import Summary
import tensorboardX as tb

import os
import json
import logging
import subprocess

logger = None


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous().to(args.device)
    logger.info("(utils) Data size: {}.".format(data.size()))
    return data


def get_batch(source, i, args, seq_len=None):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len])
    # target = Variable(source[i+1:i+1+seq_len].view(-1))
    target = Variable(source[i+1:i+1+seq_len])
    return data, target


def save_checkpoint(model, optimizer, args, finetune=False):
    model_name, opt_name = ('finetune_model.pt', 'finetune_optimizer.pt') \
        if finetune else ('model.pt', 'optimizer.pt')
    torch.save(model, os.path.join(args.model_dir, model_name))
    torch.save(optimizer.state_dict(), os.path.join(args.model_dir, opt_name))


def init_device(args):
    """Sets the `args.device` attribute based on `args.no_cuda` and host
    availability. It also sets the CUDA seed if needed.
    """
    if torch.cuda.is_available():
        if args.no_cuda:
            logger.info('WARNING: You have a CUDA device,'
                        'so you should probably not run with --no-cuda')
            setattr(args, 'device', 'cpu')
        else:
            if hasattr(args, 'seed'):
                torch.cuda.manual_seed(args.seed)
            setattr(args, 'device', args.cuda_device)
    else:
        if not args.no_cuda:
            logger.info('WARNING: No CUDA device found, using CPU. '
                        'It would be best to explicitly run with --no-cuda')
        setattr(args, 'device', 'cpu')
    logger.info("Using device: {}".format(args.device))


def prepare_dir(args):
    """Sets a model_dir if not specified in args and creates it."""
    if args.model_dir is None:
        model_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.model_dir_prefix is not None:
            model_dir = "{}-{}".format(args.model_dir_prefix, model_dir)
        model_dir = os.path.join(args.log_dir, model_dir)
        setattr(args, "model_dir", model_dir)
    if not os.path.exists(args.model_dir):
        print("Creating {}".format(args.model_dir))
        os.makedirs(args.model_dir)


def save_commit_id(args, prefix=""):
    """Saves the commit ID and the repo origin if using git."""
    # Contains byte output
    cmd = subprocess.run(["git", "rev-parse", "HEAD"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    cmd2 = subprocess.run(["git", "config", "--get", "remote.origin.url"],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if cmd.returncode == 0 and cmd.returncode == 0:
        id = cmd.stdout
        infos = cmd2.stdout
    else:
        logger.info("WARNING: Won't save GIT infos, commands return non-zero")
        return

    id_path = os.path.join(args.model_dir, prefix+"git.txt")
    with open(id_path, 'wb') as fp:
        fp.write(b"Commit ID used, GIT infos\n\n")
        fp.write(id)
        fp.write(infos)


def save_args(args, prefix=""):
    param_path = os.path.join(args.model_dir, prefix + "params.json")
    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)


def save_tb(tb, name, x, y):
    """Saves a tensorboard sacalar as a (x,y) coordinate."""
    tb.scalar_summary(name, y, x)


def save_hist(tb, name, epoch, values):
    """Saves a tensorboard histogram."""
    tb.hist_summary(name, values, epoch)


class TensorBoard(object):
    def __init__(self, model_dir, sub_dir=""):
        self.summary_writer = tb.FileWriter(os.path.join(model_dir, sub_dir))

    def scalar_summary(self, tag, value, step):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
        self.summary_writer.add_summary(summary, global_step=step)

    def hist_summary(self, tag, value, step):
        hist_summary = tb.summary.histogram(tag, value, "auto")
        self.summary_writer.add_summary(hist_summary, global_step=step)


def get_logger(args, name=__name__, level=logging.INFO, filename="file.log"):
    logger = logging.getLogger(name)
    if getattr(logger, '_init_done__', None):
        logger.setLevel(level)
        return logger
    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(0)
    f_handler = logging.FileHandler(os.path.join(args.model_dir, filename))
    f_handler.setFormatter(formatter)
    f_handler.setLevel(0)
    del logger.handlers[:]
    logger.addHandler(handler)
    logger.addHandler(f_handler)
    return logger


def set_utils_logger(my_logger):
    global logger
    logger = my_logger
