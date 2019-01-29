from common.excavator import DataSelector
import awd.embedding_mul as embedding_mul
from common.oracle import StatsKeeper
from shutil import copyfile
import torch.nn as nn
import numpy as np
import argparse
import hashlib
import torch
import time
import math
import sys
import os

import awd.model as model
from main_run import AWD

from common.utils import repackage_hidden, save_tb, save_hist

launcher = AWD()

args = launcher.args

args.tied = True
if args.size_bdrop == -1:
    args.size_bdrop = np.inf
if args.policy_every == -1:
    args.policy_every = np.inf

logger = launcher.logger
# Tensorboard
tb = launcher.tb
# DataSelector
ds = launcher.ds
# Stats Keeper
sk = launcher.sk

train_seq = launcher.train_seq

# Check if --save-grad and --embed-func == mmul
if args.save_grad and args.embed_func != "mmul":
    raise Exception(f"To use '--save-grad' '--embed-func' needs to be 'mmul'")

# Type of embedding function used
if args.embed_func == "original":
    embed_func = nn.functional.embedding
    embed = None
elif args.embed_func == "mmul":
    embedding_mul.set_logger(logger)
    embed = embedding_mul.EmbeddingMul(ds.ntokens, args.device)
    embed_func = embed

###############################################################################
# Load data
###############################################################################


def model_save(fn):
    with open(os.path.join(args.model_dir, fn), 'wb') as f:
        torch.save([model, criterion, optimizer], f)


def model_load(fn, with_copy=False):
    fn = os.path.join(args.model_dir, fn)
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f, map_location=args.device)
    if with_copy:
        copyfile(fn, os.path.join(args.model_dir, "model_loaded.pt"))


###############################################################################
# Build the model
###############################################################################

from awd.splitcross import SplitCrossEntropyLoss
criterion = None

ntokens = ds.ntokens
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers,
                       args.dropout, args.dropouth, args.dropouti, args.dropoute,
                       args.wdrop, args.tied, embed_func)
###
if args.resume:
    logger.info('Resuming model ...')
    model_load(args.resume, with_copy=True)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop:
                rnn.dropout = args.wdrop
            elif rnn.zoneout > 0:
                rnn.zoneout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    logger.info(f'Using {str(splits)}')
    criterion = SplitCrossEntropyLoss(
        args.emsize, splits=splits, verbose=False)
###
model = model.to(args.device)
criterion = criterion.to(args.device)
tot_steps = 0
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.numel() for x in params)
logger.info(f'Args: {args}')
logger.info(f'Model total parameters: {total_params}')

###############################################################################
# Training code
###############################################################################


def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN':
        model.reset()
    total_loss = 0
    ntokens = ds.ntokens
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = ds.get_batch(data_source, i)
        targets = targets.view(-1)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight,
                                            model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def evaluate_scores(epoch, batch_size):
    model.eval()
    if args.model == 'QRNN':
        model.reset()
    total_loss = 0
    ntokens = ds.ntokens
    hidden = model.init_hidden(batch_size)
    for data, targets in ds.train_seq():
        targets = targets.view(-1).contiguous()
        output, hidden = model(data, hidden)
        loss = criterion(model.decoder.weight,
                         model.decoder.bias, output, targets).data
        sk.add_prior_sample(epoch, loss.item())
        total_loss += len(data) * loss
        hidden = repackage_hidden(hidden)
    sk.save_prior_epoch()
    return total_loss.item() / ds.data_size


def train(epoch):
    global tot_steps
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN':
        model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = ds.ntokens
    hidden = model.init_hidden(ds.batch_size)
    batch, i = 0, 0

    if (epoch % args.grad_interval == 0 or epoch == 1) and \
            (args.save_grad or args.save_gradPure):
        embed.requires_grad = True
        save_grad, save_gradPure = args.save_grad, args.save_gradPure
    else:
        if embed and embed.requires_grad:
            embed.requires_grad = False
        save_grad, save_gradPure = False, False

    for data, targets in train_seq():
        # shape of data is (bptt, batch_size)
        targets = targets.view(-1).contiguous()
        seq_len = args.bptt
        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(
            data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight,
                             model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha:
            loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean()
                              for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta:
            loss = loss + \
                sum(args.beta * (rnn_h[1:] - rnn_h[:-1]
                                 ).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward(retain_graph=args.save_gradPure)
        sk.add_sample(epoch, i, loss.item())

        if save_grad:
            with torch.no_grad():
                # shape(btpp, batch_size, voc_size)
                grad = embed.last_oh.grad
                # shape(bptt, batch_size, 1, embed_size)
                res = torch.stack([
                    torch.stack([
                        torch.mm(grad[token_i, batch_i].view(1, -1), 1 /
                                 (embed.last_weight *
                                  args.emsize + sys.float_info.epsilon))
                        for batch_i in range(args.batch_size)
                    ], dim=0)
                    for token_i in range(args.bptt)
                ], dim=0)
                assert list(res.shape) == [args.bptt,
                                           args.batch_size, 1, args.emsize]
                sk.add_data("grad", epoch, i,
                            res.detach().cpu().numpy().tolist())

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip:
            torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            ppl = math.exp(cur_loss)
            bpc = cur_loss / math.log(2)
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                            epoch, i, ds.nbatch,
                            optimizer.param_groups[0]['lr'],
                            elapsed * 1000 / args.log_interval, cur_loss, ppl, bpc))
            save_tb(tb, "train/loss", tot_steps, cur_loss)
            save_tb(tb, "train/ppl", tot_steps, ppl)
            total_loss = 0
            start_time = time.time()
        ###

        ###
        if save_gradPure:
            optimizer.zero_grad()
            embed.last_oh.grad.zero_()
            output.sum().backward()
            with torch.no_grad():
                # shape(btpp, batch_size, voc_size)
                grad = embed.last_oh.grad
                res = torch.stack([
                    torch.stack([
                        torch.mm(grad[token_i, batch_i].view(1, -1), 1 /
                                 (embed.last_weight *
                                  args.emsize + sys.float_info.epsilon))
                        for batch_i in range(args.batch_size)
                    ], dim=0)
                    for token_i in range(args.bptt)
                ], dim=0)
                sk.add_data("gradPure", epoch, i,
                            res.detach().cpu().numpy().tolist())

        tot_steps += 1
        i += 1

        if tot_steps in args.when_steps:
            logger.info(f'(Step {tot_steps}) Saving model before learning '
                        'rate decreased')
            model_save('{}.e{}'.format("model.pt", epoch))
            logger.info('Dividing learning rate by 10')
            optimizer.param_groups[0]['lr'] /= 10.

        if tot_steps >= args.max_steps:
            logger.info(f"Reached max-steps at tot step {tot_steps}, breaking "
                        "the train function")
            break


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            params, lr=args.lr, weight_decay=args.wdecay)

    if args.get_priors:
        logger.info("Computing priors")
        loss = evaluate_scores(1, ds.batch_size)
        logger.info(f"Evaluated scores ({loss})")

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()

        if epoch in args.fixedbsize_epochs:
            ds.current_seq = ds.manual_seq(1)
            logger.info("Computing priors")
            loss = evaluate_scores(epoch, ds.batch_size)
            logger.info(f"Evaluated scores ({loss})")
            ds.set_fixed_bsize_seq(sk.prior_scores, args.batch_size)

        if args.update_random_rotate:
            ds.update_random_rotate_train_seq()

        train(epoch)

        # Policy stuff
        sk.save_seq(ds.current_seq)
        if epoch % args.policy_every == 0:
            ds.update_policy(np.array(sk.past_data), np.array(sk.current_data))
            args.policy_every *= args.policy_retarder
        if epoch in args.bdrop_epochs:
            ds.drop_policy(np.array(sk.past_data), np.array(sk.current_data))

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
                        'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                            epoch, epoch_time, val_loss2, ppl, val_loss2 / math.log(2)))
            logger.info('-' * 89)
            save_tb(tb, "val/loss", epoch, val_loss2)
            save_tb(tb, "val/ppl", epoch, ppl)

            if val_loss2 < stored_loss:
                model_save("model.pt")
                logger.info('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(ds.val_data, args.eval_batch_size)
            ppl = math.exp(val_loss)
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                            epoch, epoch_time, val_loss, ppl, val_loss / math.log(2)))
            logger.info('-' * 89)
            save_tb(tb, "val/loss", epoch, val_loss)
            save_tb(tb, "val/ppl", epoch, ppl)

            if val_loss < stored_loss:
                model_save("model.pt")
                logger.info('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                logger.info('Switching to ASGD')
                optimizer = torch.optim.ASGD(
                    model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                logger.info('Saving model before learning rate decreased')
                model_save('{}.e{}'.format("model.pt", epoch))
                logger.info('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)
        if tot_steps >= args.max_steps:
            logger.info("Reached max-steps, breaking the epoch loop")
            break
    sk.end()

except KeyboardInterrupt:
    logger.info('-' * 89)
    logger.info('Exiting from training early')

# Load the best saved model.
model_load("model.pt")

# Run on test data.
test_loss = evaluate(ds.test_data, args.test_batch_size)
ppl = math.exp(test_loss)
save_tb(tb, "test/loss", 1, test_loss)
save_tb(tb, "test/ppl", 1, ppl)
logger.info('=' * 89)
logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, ppl, test_loss / math.log(2)))
logger.info('=' * 89)
