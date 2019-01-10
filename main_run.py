import numpy as np
import torch
import argparse
from common.utils import prepare_dir, get_logger, set_utils_logger, init_device, save_args, save_commit_id, TensorBoard
from common.excavator import DataSelector


def add_common_args(parser, model_name):
    parser.add_argument("--main-model", type=str, required=True,
                        choices=["simple-lstm", "awd-lstm", "mos-lstm"],
                        help="The main model to use.")

    # Data, seed
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
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

    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--lr', type=float, default=30,
                        help='initial learning rate')

    # Data selection
    parser.add_argument('--init-seq', type=str, default="original",
                        help='Initialization of the ds.current_seq '
                        '(original, overlapC_N (contiguous)')


def common_init(that):
    that.args = that.init_args()

    if that.args.continue_train and that.args.model_dir is None:
        raise Exception("'--model-dir' must be specified when using "
                        "'--continue-train'")

    prepare_dir(that.args)
    that.logger = get_logger(that.args)
    set_utils_logger(that.logger)
    np.random.seed(that.args.seed)
    torch.manual_seed(that.args.seed)
    init_device(that.args)
    save_args(that.args)
    save_commit_id(that.args)
    that.tb = TensorBoard(that.args.model_dir)
    that.ds = DataSelector(that.args)

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
    else:
        raise Exception(f"init-seq unkown: {that.args.init_seq}")


class Simple:
    """This class handles the arguments"""

    def __init__(self):
        common_init(self)

    def init_args(self):
        parser = argparse.ArgumentParser(
            description='PyTorch PennTreeBank/WikiText2 RNN/LSTM Language Model')
        add_common_args(parser, "simple")
        parser.add_argument('--model', type=str, default='LSTM',
                            help='type of recurrent net '
                            '(RNN_TANH, RNN_RELU, LSTM, GRU, SRU)')

        args = parser.parse_args()
        return args


class MOS:
    """This class handles the arguments"""

    def __init__(self):
        common_init(self)

    def init_args(self):
        parser = argparse.ArgumentParser(
            description='PyTorch PennTreeBank/WikiText2 RNN/LSTM Language Model')
        add_common_args(parser, "mos")
        parser.add_argument('--model', type=str, default='LSTM',
                            help='type of recurrent net '
                            '(RNN_TANH, RNN_RELU, LSTM, GRU, SRU)')
        parser.add_argument('--nhidlast', type=int, default=-1,
                            help='number of hidden units for the last rnn layer')
        parser.add_argument('--clip', type=float, default=0.25,
                            help='gradient clipping')

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
