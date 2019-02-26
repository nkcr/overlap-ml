from main_run import Emotions
import torch
import torch.nn as nn
import numpy as np
import math
import time
from common.utils import save_tb, save_checkpoint, repackage_hidden
import os
from emotions.data import DataHandler
from collections import Counter
import operator

launcher = Emotions()
args = launcher.args

logger = launcher.logger
# Tensorboard
tb = launcher.tb
# Data Handler
dh = DataHandler(args)


class SimpleLSTM(nn.Module):
    def __init__(self, args, num_class, num_features):
        super(SimpleLSTM, self).__init__()
        self.args = args
        self.num_class = num_class
        self.num_features = num_features

        self.dropout = nn.Dropout(1 - self.args.dropout)

        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=args.nhid, num_layers=args.nlayers)

        self.decoder = nn.Linear(args.nhid, num_class)
        
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.decoder.bias.data.fill_(0.0)
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, data, hidden):
        batch_size = data.size(1)
        # hidden is a tuple (h_0, c_0)
        _, hidden = self.lstm(data, hidden)
        h_n = hidden[0][-1]
        output = self.dropout(h_n)
        output = self.decoder(output.view(-1, self.args.nhid))
        output = output.view(-1, batch_size, self.num_class)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (torch.tensor(weight.new(self.args.nlayers, batch_size,
                                        self.args.nhid).zero_()),
                torch.tensor(weight.new(self.args.nlayers, batch_size,
                                        self.args.nhid).zero_()))

def compute_accuracies(index_dict):
    results_list = list(map(lambda x: compute_result(x['class_probs'], x['target']), index_dict.values()))
    ids_acc = sum([x[0] for x in results_list]) / len(results_list)
    uw_ids_acc_dict = {}
    counter = {}
    for x in results_list:
        t = x[1]
        if t in uw_ids_acc_dict:
            uw_ids_acc_dict[t] += x[0]
            counter[t] += 1
        else:
            uw_ids_acc_dict[t] = x[0]
            counter[t] = 1
    
    for k,v in counter.items():
        uw_ids_acc_dict[k] /= v

    uw_ids_acc = sum([v for k,v in uw_ids_acc_dict.items()]) / len(counter.keys())

    return ids_acc, uw_ids_acc

def compute_result(class_probs, target):
    
    # Sum all the probabilities and choose as prediction the highest one
    unweighted_sum = {k: sum([c[k] for c in class_probs]) for k in class_probs[0].keys()}
    pred = max(unweighted_sum.items(), key=operator.itemgetter(1))[0]
    res = 1 if pred == target else 0
    return res, target

def init_model(model):
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda(args.device)
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            nn.init.xavier_uniform_(p)
    return model

def evaluate(dh, batch_size=10):
    model.eval()
    total_loss = 0
    n_steps = 0
    hidden = model.init_hidden(batch_size)
    index_dict = {}
    with torch.no_grad():
        
        for data, targets, id_ in dh.test_seq():
            n_steps += 1
            data = data.permute(1,0,2)
            remove_padding = False
            remaining_samples = 0
            if data.shape[1] != args.batch_size:
                remove_padding = True
                remaining_samples = args.batch_size - data.shape[1]
                padding = torch.zeros(data.shape[0], remaining_samples, data.shape[2])
                data = torch.cat((data, padding), 1)
            if torch.cuda.is_available():
                data = data.cuda()
                targets = targets.cuda()
            targets = targets.view(-1)
            y_classes = targets.tolist()

            output, hidden = model(data, hidden)
            if remove_padding:
                output = output[:, :-remaining_samples, :]
            loss = criterion(output.view(-1, output.size(2)), targets).data
            total_loss += loss

            for target, probs, id_ in zip(y_classes, output.view(-1, output.size(2)).tolist(), id_):
                dict_ = {i:x for i,x in enumerate(probs)}
                if id_ in index_dict:
                    index_dict[id_]['class_probs'].append(Counter(dict_))                    
                else:
                    index_dict[id_] = {'class_probs': [Counter(dict_)], 'target': target}

        ids_acc, uw_ids_acc = compute_accuracies(index_dict)

    return total_loss.item() / n_steps, ids_acc, uw_ids_acc


def train():
    global tot_steps
    batch = 0
    total_loss = 0
    model.train()
    hidden = model.init_hidden(args.batch_size)
    start_time = time.time()
    for data, targets, id_ in dh.train_seq():
        data = data.permute(1,0,2)
        if data.shape[1] != args.batch_size:
            continue
        if torch.cuda.is_available():
            data = data.cuda()
            targets = targets.cuda()
        targets = targets.view(-1)
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, output.size(2)), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            # ppl = math.exp(cur_loss)
            lr = optimizer.param_groups[0]['lr']
            ms_batch = elapsed * 1000 / args.log_interval
            logger.info('| epoch {:3d} | '.format(epoch) +
                        'lr {:02.2f} | ms/batch {:5.2f} | '.format(lr, ms_batch) +
                        'loss {:5.2f}'.format(cur_loss))
            save_tb(tb, "train/loss", tot_steps, cur_loss)
            total_loss = 0
            start_time = time.time()

        batch += 1
        tot_steps += 1


if args.continue_train:
    model = torch.load(os.path.join(args.model_dir, 'model.pt'))
    logger.info("Loading 'model.pt' at {}.".format(args.model_dir))
    optimizer_state = torch.load(os.path.join(args.model_dir, 'optimizer.pt'))
    logger.info("Loading 'optimizer.pt' at {}.".format(args.model_dir))
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.lr, weight_decay=args.wdecay, momentum=args.momentum)
    optimizer.load_state_dict(optimizer_state)
else:
    model = SimpleLSTM(args, dh.num_class, dh.num_features)
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.lr, weight_decay=args.wdecay, momentum=args.momentum)

criterion = nn.CrossEntropyLoss()


model = init_model(model)
# model = model.to(args.device)
total_params = sum(x.data.nelement() for x in model.parameters())
logger.info('Args: {}'.format(args))
logger.info('Model total parameters: {}'.format(total_params))

tot_steps = 0

best_val_loss = []
stored_loss = np.inf

try:
    for epoch in range(1, args.epochs+1):

        lr_decay = args.lr_decay ** max(epoch+1 - args.lr_decay_start, 0)
        optimizer.param_groups[0]['lr'] = args.lr * lr_decay

        if epoch in args.when:
            logger.info("Dividing the loss by 10")
            optimizer.param_groups[0]['lr'] /= 10

        epoch_start_time = time.time()
        train()
        epoch_time = time.time() - epoch_start_time
        save_tb(tb, "time/epoch", epoch, epoch_time)

        val_loss, ids_acc, uw_ids_acc = evaluate(dh, args.test_batch_size)
        # ppl = math.exp(val_loss)
        logger.info('-' * 89)
        logger.info('| end of epoch {:3d} | time: {:5.2f}s | '.format(epoch, epoch_time) +
                    'valid loss {:5.2f} | '.format(val_loss) +
                    'valid ids_acc {:5.2f} | valid uw_ids_acc {:8.2f}'.format(ids_acc, uw_ids_acc))
        logger.info('-' * 89)
        save_tb(tb, "val/loss", epoch, val_loss)
        save_tb(tb, "val/ids_acc", epoch, ids_acc)
        save_tb(tb, "val/uw_ids_acc", epoch, uw_ids_acc)

        if val_loss < stored_loss:
            save_checkpoint(model, optimizer, args)
            logger.info('Saving Normal!')
            stored_loss = val_loss
        best_val_loss.append(val_loss)

except KeyboardInterrupt:
    logger.info('-' * 89)
    logger.info('Exiting from training early')

# # Load the best saved model.
# model = torch.load(os.path.join(args.model_dir, 'model.pt')).to(args.device)

# # Run on test data.
# test_loss = evaluate(ds.test_data, args.test_batch_size)
# ppl = math.exp(test_loss)
# save_tb(tb, "test/loss", 1, test_loss)
# save_tb(tb, "test/ppl", 1, ppl)
# logger.info('=' * 89)
# logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#     test_loss, ppl))
# logger.info('| Best valid | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
#     stored_loss, math.exp(val_loss)))
# logger.info('=' * 89)
