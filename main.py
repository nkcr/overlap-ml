import argparse
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import gc

import data
import model

from utils import batchify, get_batch, repackage_hidden, \
    save_checkpoint, prepare_dir, get_logger, set_utils_logger, init_device, \
    save_args, save_commit_id, TensorBoard, save_tb

parser = argparse.ArgumentParser(
    description='PyTorch PennTreeBank/WikiText2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net '
                    '(RNN_TANH, RNN_RELU, LSTM, GRU, SRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=-1,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
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
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda-device', type=str, default='cuda:0')
parser.add_argument('--no-cuda', action='store_true',
                    help='do NOT use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')

# parser.add_argument('--save', type=str,  default='EXP',
#                     help='path to save the final model')
parser.add_argument('--log-dir', type=str, default='logs',
                    help="Directory containing the runs")
parser.add_argument('--model-dir', type=str, help='Directory of the run. '
                    'If not specified, one is created based on the time.')
parser.add_argument('--model-dir-prefix', type=str,
                    help='A prefix to be added if the model-dir is '
                    'automatically created. Has no effect if --model-dir '
                    'is specified.')

parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation '
                    '(alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN '
                    'activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--n_experts', type=int, default=10,
                    help='number of experts')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should '
                    'be divisible by small_batch_size. In our implementation, '
                    'we compute gradients with small_batch_size multiple, '
                    'times and accumulate the gradients until batch_size is '
                    'reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=40,
                    help='max sequence length')
parser.add_argument('--single_gpu', default=False, action='store_true',
                    help='use single GPU')
args = parser.parse_args()

if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.dropoutl < 0:
    args.dropoutl = args.dropouth
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

# Prepare and sets the `model_dir`
prepare_dir(args)
# Logger init and set for utils
logger = get_logger(args)
set_utils_logger(logger)
# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# Sets the `args.device` and CUDA seed.
init_device(args)
# Save infos
save_args(args)
save_commit_id(args)
# Tensorboard
tb = TensorBoard(args.model_dir)


###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if args.continue_train:
    model = torch.load(os.path.join(args.model_dir, 'model.pt'))
    logger.info(f"Loading 'model.pt' at {args.model_dir}.")
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid,
                           args.nhidlast, args.nlayers, args.dropout,
                           args.dropouth, args.dropouti, args.dropoute,
                           args.wdrop, args.tied, args.dropoutl,
                           args.n_experts)

parallel_model = model.to(args.device)

total_params = sum(x.data.nelement() for x in model.parameters())
logger.info('Args: {}'.format(args))
logger.info('Model total parameters: {}'.format(total_params))

criterion = nn.CrossEntropyLoss()
tot_steps = 0

###############################################################################
# Training code
###############################################################################


def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args)
            targets = targets.view(-1)

            log_prob, hidden = parallel_model(data, hidden)
            loss = nn.functional.nll_loss(
                log_prob.view(-1, log_prob.size(2)), targets).data

            total_loss += len(data) * loss
            hidden = repackage_hidden(hidden)

    return total_loss.item() / len(data_source)


def train():
    global tot_steps
    assert args.batch_size % args.small_batch_size == 0, \
        'batch_size must be divisible by small_batch_size'

    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = [model.init_hidden(args.small_batch_size) for _ in range(
        args.batch_size // args.small_batch_size)]
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        seq_len = args.bptt

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        optimizer.zero_grad()

        start, end, s_id = 0, args.small_batch_size, 0
        while start < args.batch_size:
            cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous(
            ).view(-1)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden[s_id] = repackage_hidden(hidden[s_id])

            log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = parallel_model(
                cur_data, hidden[s_id], return_h=True)
            raw_loss = nn.functional.nll_loss(
                log_prob.view(-1, log_prob.size(2)), cur_targets)

            loss = raw_loss
            # Activiation Regularization
            loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean()
                              for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            loss = loss + \
                sum(args.beta * (rnn_h[1:] - rnn_h[:-1]
                                 ).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss *= args.small_batch_size / args.batch_size
            total_loss += raw_loss.data * args.small_batch_size / args.batch_size
            loss.backward()

            s_id += 1
            start = end
            end = start + args.small_batch_size

            gc.collect()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            ppl = math.exp(cur_loss)
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                            epoch, batch, len(train_data) // args.bptt,
                            optimizer.param_groups[0]['lr'],
                            elapsed * 1000 / args.log_interval, cur_loss, ppl))
            save_tb(tb, "train/loss", tot_steps, cur_loss)
            save_tb(tb, "train/ppl", tot_steps, ppl)
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
        tot_steps += 1


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    if args.continue_train:
        optimizer_state = torch.load(
            os.path.join(args.model_dir, 'optimizer.pt'))
        logger.info(f"Loading 'optimizer.pt' at {args.model_dir}.")
        if 't0' in optimizer_state['param_groups'][0]:
            optimizer = torch.optim.ASGD(
                model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.lr, weight_decay=args.wdecay)
        optimizer.load_state_dict(optimizer_state)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        epoch_time = time.time() - epoch_start_time
        save_tb(tb, "time/epoch", epoch, epoch_time)
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data)
            ppl = math.exp(val_loss2)
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, epoch_time,
                                                   val_loss2, ppl))
            logger.info('-' * 89)
            save_tb(tb, "val/loss", epoch, val_loss2)
            save_tb(tb, "val/ppl", epoch, ppl)

            if val_loss2 < stored_loss:
                save_checkpoint(model, optimizer, args)
                logger.info('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(val_data, eval_batch_size)
            ppl = math.exp(val_loss)
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, epoch_time,
                                                   val_loss, ppl))
            logger.info('-' * 89)
            save_tb(tb, "val/loss", epoch, val_loss)
            save_tb(tb, "val/ppl", epoch, ppl)

            if val_loss < stored_loss:
                save_checkpoint(model, optimizer, args)
                logger.info('Saving Normal!')
                stored_loss = val_loss

            if 't0' not in optimizer.param_groups[0] and (len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                logger.info('Switching!')
                optimizer = torch.optim.ASGD(
                    model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                #optimizer.param_groups[0]['lr'] /= 2.
            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    logger.info('-' * 89)
    logger.info('Exiting from training early')

# Load the best saved model.
model = torch.load(os.path.join(args.model_dir, 'model.pt'))
parallel_model = nn.DataParallel(model, dim=1).to(args.device)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
ppl = math.exp(test_loss)
save_tb(tb, "test/loss", 1, test_loss)
save_tb(tb, "test/ppl", 1, ppl)
logger.info('=' * 89)
logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, ppl))
logger.info('=' * 89)
