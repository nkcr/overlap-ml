from main_run import Simple
import torch
import torch.nn as nn
import numpy as np
import math
import time
from common.utils import save_tb, save_checkpoint, repackage_hidden
import os

launcher = Simple()
args = launcher.args

logger = launcher.logger
# Tensorboard
tb = launcher.tb
# DataSelector
ds = launcher.ds

train_seq = launcher.train_seq


class SimpleLSTM(nn.Module):
    def __init__(self, args, ntokens):
        super(SimpleLSTM, self).__init__()
        self.args = args
        self.ntokens = ntokens

        self.embedding = nn.Embedding(ntokens, args.nhid)

        self.dropout = nn.Dropout(1 - self.args.dropout)

        self.lstm = nn.LSTM(input_size=args.nhid,
                            hidden_size=args.nhid, num_layers=args.nlayers)

        self.decoder = nn.Linear(args.nhid, ntokens)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0.0)
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, data, hidden):
        batch_size = data.size(1)
        # hidden is a tuple (h_0, c_0)
        data = self.dropout(self.embedding(data))
        output, hidden = self.lstm(data, hidden)
        output = self.dropout(output)
        output = self.decoder(output.view(-1, self.args.nhid))
        output = output.view(-1, batch_size, self.ntokens)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (torch.tensor(weight.new(self.args.nlayers, batch_size,
                                        self.args.nhid).zero_()),
                torch.tensor(weight.new(self.args.nlayers, batch_size,
                                        self.args.nhid).zero_()))


if args.continue_train:
    model = torch.load(os.path.join(args.model_dir, 'model.pt'))
    logger.info(f"Loading 'model.pt' at {args.model_dir}.")
    optimizer_state = torch.load(os.path.join(args.model_dir, 'optimizer.pt'))
    logger.info(f"Loading 'optimizer.pt' at {args.model_dir}.")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    optimizer.load_state_dict(optimizer_state)
else:
    model = SimpleLSTM(args, ds.ntokens)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, weight_decay=args.wdecay)

criterion = nn.CrossEntropyLoss()

model = model.to(args.device)
total_params = sum(x.data.nelement() for x in model.parameters())
logger.info('Args: {}'.format(args))
logger.info('Model total parameters: {}'.format(total_params))

tot_steps = 0


def evaluate(data_source, batch_size=10):
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = ds.get_batch(data_source, i)
            targets = targets.view(-1)

            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, output.size(2)), targets).data
            total_loss += len(data) * loss

            hidden = repackage_hidden(hidden)

    return total_loss.item() / len(data_source)


def train():
    global tot_steps
    batch = 0
    total_loss = 0
    model.train()
    hidden = model.init_hidden(args.batch_size)
    start_time = time.time()
    for data, targets in ds.train_seq():
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
            ppl = math.exp(cur_loss)
            lr = optimizer.param_groups[0]['lr']
            ms_batch = elapsed * 1000 / args.log_interval
            logger.info(f'| epoch {epoch:3d} | '
                        f'{batch:5d}/{ds.nbatch:5d} batches | '
                        f'lr {lr:02.2f} | ms/batch {ms_batch:5.2f} | '
                        f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            save_tb(tb, "train/loss", tot_steps, cur_loss)
            save_tb(tb, "train/ppl", tot_steps, ppl)
            total_loss = 0
            start_time = time.time()

        batch += 1
        tot_steps += 1


best_val_loss = []
stored_loss = np.inf

try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        epoch_time = time.time() - epoch_start_time
        save_tb(tb, "time/epoch", epoch, epoch_time)

        val_loss = evaluate(ds.val_data, args.eval_batch_size)
        ppl = math.exp(val_loss)
        logger.info('-' * 89)
        logger.info(f'| end of epoch {epoch:3d} | time: {epoch_time:5.2f}s | '
                    f'valid loss {val_loss:5.2f} | valid ppl {ppl:8.2f}')
        logger.info('-' * 89)
        save_tb(tb, "val/loss", epoch, val_loss)
        save_tb(tb, "val/ppl", epoch, ppl)

        if val_loss < stored_loss:
            save_checkpoint(model, optimizer, args)
            logger.info('Saving Normal!')
            stored_loss = val_loss
        best_val_loss.append(val_loss)

except KeyboardInterrupt:
    logger.info('-' * 89)
    logger.info('Exiting from training early')

# Load the best saved model.
model = torch.load(os.path.join(args.model_dir, 'model.pt')).to(args.device)

# Run on test data.
test_loss = evaluate(ds.test_data, args.test_batch_size)
ppl = math.exp(test_loss)
save_tb(tb, "test/loss", 1, test_loss)
save_tb(tb, "test/ppl", 1, ppl)
logger.info('=' * 89)
logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, ppl))
logger.info('| Best valid | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
    stored_loss, math.exp(val_loss)))
logger.info('=' * 89)
