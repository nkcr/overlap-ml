import os
import numpy as np
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from collections import Counter
from nohemien_utils import load_nohemien, AudioWindowDataset, collate_fn

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

"""
Arguments: if you use a config file, put them there
"""
# Do not change this one
cv = 5
weighted = True  # unbalanced classes

# Change to your location
# data_path = 'path/to/dir'
data_path = '../../data/IEMOCAP/all_features_cv'

# Change to number of frames and step size preferred
window_size = 500  # number of frames
step_size = 0.1  # percentage of the window

# Training parameters
batch_size = 8
lr = 1e-6
opt = torch.optim.Adam
loss_function = nn.CrossEntropyLoss

# orders = ['random-no_step','order-no_step','random-step','order-step']
orders = ['complete_random', 'local_order', 'standard_order', 'total_order']
order = orders[-1]

train_features, train_labels, train_ids, test_features, test_labels, test_ids = load_nohemien(
    data_path, cv)

# Creating dataset and dataloader
train = AudioWindowDataset(train_features, train_labels, train_ids,
                           window_size=window_size, step_size=step_size, how=order, batch_size=batch_size)
data_loader_train = DataLoader(train, collate_fn=collate_fn,
                               batch_size=batch_size, num_workers=0, sampler=None, shuffle=False)
test = AudioWindowDataset(test_features, test_labels, test_ids,
                          window_size=window_size, step_size=step_size, how=order, batch_size=batch_size)
data_loader_test = DataLoader(
    test, collate_fn=collate_fn, batch_size=batch_size, num_workers=0)

weights = None
if weighted:
    counter = Counter(train_labels.numpy())
    min_ = min(counter.values())
    weights = torch.FloatTensor([1 / (counter[v] / min_)
                                 for v in range(len(counter))])

# TODO: build your model here
model = None  # Build your model
# TODO: once you built your model, uncomment line below
# loss_compute = SimpleLossCompute(loss_function(weight=weights), opt=opt(model.parameters(), lr=lr))

print(train.order[:batch_size])

for i, batch in enumerate(data_loader_train):

    batch_x = Variable(batch.src, requires_grad=False)
    batch_y = Variable(batch.trg, requires_grad=False)
    print(batch_x.shape)
    print(batch_y.shape)
    print(batch.ids)
    break
