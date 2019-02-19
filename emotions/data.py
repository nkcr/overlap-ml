import os
import numpy as np
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from collections import Counter
from emotions.utils import load, AudioWindowDataset, collate_fn


class DataHandler:

    def __init__(self, args):
        self.cv = args.cv
        self.weighted = args.weighted
        self.args = args
        self.data_path = args.data
        self.window_size = args.window_size
        self.step_size = args.step_size
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.lr = args.lr
        self.num_features = 384  # to be refactored..

        if args.optimizer == 'sgd':
            self.opt = torch.optim.SGD
        if args.optimizer == 'adam':
            self.opt = torch.optim.Adam

        self.loss_function = nn.CrossEntropyLoss

        self.order = args.order

        self.load_data()

    def train_seq(self):
        for i, batch in enumerate(self.train_data):
            batch_x = Variable(batch.src, requires_grad=False)
            batch_y = Variable(batch.trg, requires_grad=False)
            id_ = batch.ids
            yield batch_x, batch_y, id_
    
    def test_seq(self):
        for i, batch in enumerate(self.test_data):
            batch_x = Variable(batch.src, requires_grad=False)
            batch_y = Variable(batch.trg, requires_grad=False)
            id_ = batch.ids
            yield batch_x, batch_y, id_

    def load_data(self):
        train_features, train_labels, train_ids, test_features,\
            test_labels, test_ids, self.num_class = load(
                self.data_path, self.cv)

        # Creating dataset and dataloader
        train = AudioWindowDataset(train_features, train_labels, train_ids,
                                   window_size=self.window_size,
                                   step_size=self.step_size, how=self.order,
                                   batch_size=self.batch_size)
        self.train_data = DataLoader(train, collate_fn=collate_fn,
                                     batch_size=self.batch_size,
                                     num_workers=0, sampler=None,
                                     shuffle=False)
        test = AudioWindowDataset(test_features, test_labels, test_ids,
                                  window_size=self.window_size,
                                  step_size=self.step_size,
                                  how=self.order, batch_size=self.test_batch_size)
        self.test_data = DataLoader(test, collate_fn=collate_fn,
                                    batch_size=self.test_batch_size,
                                    num_workers=0)

        self.weights = None
        if self.weighted:
            counter = Counter(train_labels.numpy())
            min_ = min(counter.values())
            self.weights = torch.FloatTensor([1 / (counter[v] / min_)
                                              for v in range(len(counter))])

    # TODO: build your model here
    # model = None  # Build your model
    # TODO: once you built your model, uncomment line below
    # loss_compute = SimpleLossCompute(loss_function(weight=weights), opt=opt(model.parameters(), lr=lr))

    # print(train.order[:batch_size])
