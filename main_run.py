import numpy as np
import torch
import argparse
from common.utils import prepare_dir, get_logger, set_utils_logger, init_device, save_args, save_commit_id, TensorBoard
from common.excavator import DataSelector
from common.oracle import StatsKeeper
import time
import random


def add_common_args(parser, model_name):
    parser.add_argument("--main-model", type=str, required=True,
                        choices=["simple-lstm", "awd-lstm", "mos-lstm"],
                        help="The main model to use.")

    # Data, seed
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--seed-shuffle', type=int, default=141,
                        help='Seed for the batch shuffling.')
    parser.add_argument('--data', type=str, default='data/penn/',
                        help='location of the data corpus')
    parser.add_argument('--emsize', type=int, default=400,
                        help='size of word embeddings')

    # Cuda
    parser.add_argument('--cuda-device', type=str, default='cuda:0')
    parser.add_argument('--no-cuda', action='store_true',
                        help='do NOT use CUDA')

    # log directory, log interval
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--log-dir', type=str, default=f'logs/{model_name}',
                        help="Directory containing the runs")
    parser.add_argument('--model-dir', type=str, help='Directory of the run. '
                        'If not specified, one is created based on the time.')
    parser.add_argument('--model-dir-prefix', type=str,
                        help='A prefix to be added if the model-dir is '
                        'automatically created. Has no effect if --model-dir '
                        'is specified.')
    parser.add_argument('--continue-train', action='store_true',
                        help='continue train from a checkpoint')

    # epochs, batch sizes, bptt
    parser.add_argument('--epochs', type=int, default=8000,
                        help='upper epoch limit')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--eval-batch-size', type=int, default=10, metavar='N',
                        help='eval batch size')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='test batch size')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')

    parser.add_argument('--nhid', type=int, default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')

    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--lr', type=float, default=30,
                        help='initial learning rate')

    # Data selection
    parser.add_argument('--init-seq', type=str, default="original",
                        help='Initialization of the ds.current_seq '
                        '(original, overlapC_N (contiguous), '
                        'overlapCN_N (contiguous normalized')
    parser.add_argument('--train-seq', type=str, default="original",
                        help='Which ds.train_seq method to use '
                        '(original, repeat_N')
    parser.add_argument('--stat-folder', type=str, default="stats/",
                        help='Folder to store the stats inside the log '
                        'folder if relative, else to the absolute path plus '
                        'the model id.')

    # Train seq shuffling
    parser.add_argument('--shuffle-row-seq', action="store_true",
                        help="Shuffles the ds.train_seq row-wise before training")
    parser.add_argument('--shuffle-col-seq', action="store_true",
                        help="Shuffles the ds.train_seq column-wise before training")
    parser.add_argument('--shuffle-each-row-seq', action="store_true",
                        help="Shuffles the ds.train_seq for each row individually")
    parser.add_argument('--shuffle-full-seq', action="store_true",
                        help="Shuffles the ds.train_seq row and column wise (complete)")


def common_init(that):
    """Common initialization of our models. Here is the check list:

        - [√] Parse the input arguments
        - [√] Create necessary folders to save data
        - [√] Set a logger to be used and save the output 
        - [√] Set manual seeds to make results reproductible
        - [√] Init the correct device to be used by pytorch: cpu or cuda:id
        - [√] Save the input arguments used
        - [√] Save the git infos: commit id, repo origin
        - [√] Set a tensorboard object to record stats
        - [√] Set a DataSelector object which handles data samples
        - [√] Set a StatKeeper object which can save arbitrary stats
        - [√] Perform specific initializations based on input params
    """
    that.args = that.init_args()

    if that.args.continue_train and that.args.model_dir is None:
        raise Exception("'--model-dir' must be specified when using "
                        "'--continue-train'")

    prepare_dir(that.args)
    that.logger = get_logger(that.args)
    set_utils_logger(that.logger)
    np.random.seed(that.args.seed)
    random.seed(that.args.seed)
    torch.manual_seed(that.args.seed)
    init_device(that.args)
    save_args(that.args)
    save_commit_id(that.args)
    that.tb = TensorBoard(that.args.model_dir)
    that.ds = DataSelector(that.args)
    that.sk = StatsKeeper(that.args, that.args.stat_folder)

    # Init seq
    if that.args.init_seq == "original":
        # Done by default in DataSelector initialization
        pass
    elif that.args.init_seq.startswith("overlapC_"):
        overlap = int(that.args.init_seq.split("_")[1])
        if that.args.bptt % overlap != 0:
            raise Exception(f"overlapC must divide '--bptt' (found {overlap})")
        that.ds.current_seq = that.ds.overlap_c_seq(
            that.args.batch_size, overlap)
    elif that.args.init_seq.startswith("overlapCN_"):
        overlap = int(that.args.init_seq.split("_")[1])
        if that.args.bptt % overlap != 0:
            raise Exception(
                f"overlapCN must divide '--bptt' (found {overlap})")
        that.ds.current_seq = that.ds.overlap_cn_seq(
            that.args.batch_size, overlap)
    else:
        raise Exception(f"init-seq unkown: {that.args.init_seq}")

    # Type of train_seq
    if that.args.train_seq == "original":
        that.train_seq = that.ds.train_seq
    elif that.args.train_seq.startswith("repeat_"):
        n = int(that.args.train_seq.split("_")[1])
        that.train_seq = lambda: that.ds.repeated_train_seq(n)
    else:
        raise Exception(f"train-seq unkown: {that.args.train_seq}")

    # Shuffling of the train_seq
    if that.args.shuffle_row_seq:
        that.ds.shuffle_row_train_seq()
    if that.args.shuffle_col_seq:
        that.ds.shuffle_col_train_seq()
    if that.args.shuffle_each_row_seq:
        that.ds.shuffle_each_row_train_seq()
    if that.args.shuffle_full_seq:
        that.ds.shuffle_full_train_seq()


class Simple:
    """This class handles the arguments"""

    def __init__(self):
        common_init(self)

    def init_args(self):
        parser = argparse.ArgumentParser(
            description='PyTorch PennTreeBank/WikiText2 RNN/LSTM Language Model',
            conflict_handler='resolve', allow_abbrev=False)
        add_common_args(parser, "simple")

        parser.add_argument('--dropout', type=float,
                            default=0.35, help="Probability to keep")
        parser.add_argument('--momentum', type=float,
                            default=0.0, help="Momentum of the optimizer")
        parser.add_argument('--lr-decay', type=float,
                            default=0.87, help="Decay of learning rate")
        parser.add_argument('--lr-decay-start', type=int,
                            default=0, help="Epochs when lr decay starts")

        args = parser.parse_args()
        return args


class AWD:
    """This class handles the arguments"""

    def __init__(self):
        common_init(self)

    def init_args(self):
        parser = argparse.ArgumentParser(
            description='PyTorch PennTreeBank/WikiText2 RNN/LSTM Language Model',
            conflict_handler='resolve', allow_abbrev=False)

        add_common_args(parser, "awd")

        parser.add_argument('--model', type=str, default='LSTM',
                            help='type of recurrent net (LSTM, QRNN, GRU)')
        parser.add_argument('--dropout', type=float, default=0.4,
                            help='dropout applied to layers (0 = no dropout)')
        parser.add_argument('--dropouth', type=float, default=0.3,
                            help='dropout for rnn layers (0 = no dropout)')
        parser.add_argument('--dropouti', type=float, default=0.65,
                            help='dropout for input embedding layers (0 = no dropout)')
        parser.add_argument('--dropoute', type=float, default=0.1,
                            help='dropout to remove words from embedding layer (0 = no dropout)')
        parser.add_argument('--wdrop', type=float, default=0.5,
                            help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
        parser.add_argument('--nonmono', type=int, default=5,
                            help='random seed')
        parser.add_argument('--alpha', type=float, default=2,
                            help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
        parser.add_argument('--beta', type=float, default=1,
                            help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
        parser.add_argument('--wdecay', type=float, default=1.2e-6,
                            help='weight decay applied to all weights')
        parser.add_argument('--resume', type=str,  default='',
                            help='path of model to resume')
        parser.add_argument('--optimizer', type=str,  default='sgd',
                            help='optimizer to use (sgd, adam)')
        parser.add_argument('--when', nargs="+", type=int, default=[-1],
                            help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

        # Our custum parameters
        parser.add_argument('--batch-max', type=int, default=128,
                            help='Maximum number of data point in a batch')
        parser.add_argument('--policy-every', type=int, default=-1,
                            help='Steps when to update the policy')
        parser.add_argument('--policy-retarder', type=int, default=1,
                            help='Multiplicator of the --policy-every')
        parser.add_argument('--size-bdrop', type=int, default="-1",
                            help='Which batch size we drop half batch')
        parser.add_argument('--bdrop-epochs', nargs="+", type=int, default=[-1],
                            help='Which epoch we drop half batch')
        parser.add_argument('--init-seq', type=str, default="original",
                            help='Initialization of the ds.current_seq '
                            '(original, one, overlap_2, overlapC_2 (contiguous), '
                            'overlapCP_2 (contiguous-pruned), overlapCR_2 (contiguous-row), '
                            'overlapCF_2 (contiguous-fake), rotate_2, random_rotate, transposed')
        parser.add_argument('--get-priors', action="store_true",
                            help="Computes loss of each batch")

        parser.add_argument('--update-random-rotate', action="store_true",
                            help="Rotates the current_seq randomly at each epoch using seed-shuffle")
        parser.add_argument('--fixedbsize-epochs', nargs="+", type=int, default=[-1],
                            help='Which epoch we fill the batches according to scores')
        parser.add_argument('--fixedbsize-policy', type=str, default="original",
                            help='Policy of the fixed batch size. By default it sorts using the score '
                            'Possible values: (original, combined)')
        parser.add_argument('--window-end', type=int, default="-1",
                            help='At which epoch the window reaches the end of every batches')
        parser.add_argument('--reverse-score', action="store_true",
                            help="Instead of selecting higher is better, selects lower is better")
        parser.add_argument('--shuffle-chunks', type=int, default=-1,
                            help="Shuffles the train tokens by N chunks")
        parser.add_argument('--shuffle-chunks-size', type=int, default=-1,
                            help="Shuffles the train tokens by chunks of N size")
        parser.add_argument('--embed-func', type=str, default='original',
                            help="Type of embedding function used", choices=['original', 'mmul'])
        parser.add_argument('--save-grad', action="store_true",
                            help="Saves the grd wrt to the input.")
        parser.add_argument('--save-gradPure', action="store_true",
                            help="Saves the grad wrt to the input but backwarding the output, not the loss.")
        parser.add_argument("--grad-interval", type=int, default=20,
                            help="Which epoch interval we save the grads.")

        args = parser.parse_args()
        return args


class MOS:
    """This class handles the arguments"""

    def __init__(self):
        common_init(self)

    def init_args(self):
        parser = argparse.ArgumentParser(
            description='PyTorch PennTreeBank/WikiText2 RNN/LSTM Language Model',
            conflict_handler='resolve', allow_abbrev=False)
        add_common_args(parser, "mos")
        parser.add_argument('--model', type=str, default='LSTM',
                            help='type of recurrent net '
                            '(RNN_TANH, RNN_RELU, LSTM, GRU, SRU)')
        parser.add_argument('--nhidlast', type=int, default=-1,
                            help='number of hidden units for the last rnn layer')

        parser.add_argument('--dropout', type=float, default=0.4,
                            help='dropout applied to layers (0 = no dropout)')
        parser.add_argument('--dropouth', type=float, default=0.3,
                            help='dropout for rnn layers (0 = no dropout)')
        parser.add_argument('--dropouti', type=float, default=0.65,
                            help='dropout for input embedding layers (0 = no dropout)')
        parser.add_argument('--dropoute', type=float, default=0.1,
                            help='dropout to remove words from embedding layer '
                            '(0 = no dropout)')
        parser.add_argument('--dropoutl', type=float, default=-0.2,
                            help='dropout applied to layers (0 = no dropout)')
        parser.add_argument('--wdrop', type=float, default=0.5,
                            help='amount of weight dropout to apply to the RNN hidden '
                            'to hidden matrix')
        parser.add_argument('--tied', action='store_false',
                            help='tie the word embedding and softmax weights')
        parser.add_argument('--nonmono', type=int, default=5,
                            help='random seed')

        parser.add_argument('--alpha', type=float, default=2,
                            help='alpha L2 regularization on RNN activation '
                            '(alpha = 0 means no regularization)')
        parser.add_argument('--beta', type=float, default=1,
                            help='beta slowness regularization applied on RNN '
                            'activiation (beta = 0 means no regularization)')
        parser.add_argument('--n-experts', type=int, default=10,
                            help='number of experts')
        parser.add_argument('--small-batch-size', type=int, default=-1,
                            help='the batch size for computation. batch_size should '
                            'be divisible by small_batch_size. In our implementation, '
                            'we compute gradients with small_batch_size multiple, '
                            'times and accumulate the gradients until batch_size is '
                            'reached. An update step is then performed.')
        parser.add_argument('--max-seq-len-delta', type=int, default=40,
                            help='max sequence length')
        parser.add_argument('--single-gpu', default=False, action='store_true',
                            help='use single GPU')

        args = parser.parse_args()
        return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse only the main model argument")
    parser.add_argument("--main-model", type=str, required=True,
                        choices=["simple-lstm", "awd-lstm", "mos-lstm"],
                        help="The main model to use.")
    args, remaining = parser.parse_known_args()

    if args.main_model == "simple-lstm":
        from simple import main
    elif args.main_model == "mos-lstm":
        from mos import main
    elif args.main_model == "awd-lstm":
        from awd import main
