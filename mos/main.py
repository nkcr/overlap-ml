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

import common.data
from mos import model
from common.excavator import DataSelector

from common.utils import repackage_hidden, \
    save_checkpoint, prepare_dir, get_logger, set_utils_logger, init_device, \
    save_args, save_commit_id, TensorBoard, save_tb

from main_run import MOS

launcher = MOS()

args = launcher.args

if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.dropoutl < 0:
    args.dropoutl = args.dropouth
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size


###############################################################################
# Initializations
###############################################################################

logger = launcher.logger
# Tensorboard
tb = launcher.tb
# DataSelector
ds = launcher.ds

train_seq = launcher.train_seq

###############################################################################
# Build the model
###############################################################################

if args.continue_train:
    model = torch.load(os.path.join(args.model_dir, 'model.pt'))
    logger.info(f"Loading 'model.pt' at {args.model_dir}.")
else:
    model = model.RNNModel(args.model, ds.ntokens, args.emsize, args.nhid,
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
    hidden = model.init_hidden(batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = ds.get_batch(data_source, i)
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
    hidden = [model.init_hidden(args.small_batch_size) for _ in range(
        args.batch_size // args.small_batch_size)]
    batch = 0
    for data, targets in train_seq():
        seq_len = args.bptt

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()

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
                            epoch, batch, len(ds.current_seq),
                            optimizer.param_groups[0]['lr'],
                            elapsed * 1000 / args.log_interval, cur_loss, ppl))
            save_tb(tb, "train/loss", tot_steps, cur_loss)
            save_tb(tb, "train/ppl", tot_steps, ppl)
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
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

            val_loss2 = evaluate(ds.val_data)
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
            val_loss = evaluate(ds.val_data, args.eval_batch_size)
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
test_loss = evaluate(ds.test_data, args.test_batch_size)
ppl = math.exp(test_loss)
save_tb(tb, "test/loss", 1, test_loss)
save_tb(tb, "test/ppl", 1, ppl)
logger.info('=' * 89)
logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, ppl))
logger.info('=' * 89)
