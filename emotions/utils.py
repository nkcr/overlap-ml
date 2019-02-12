from torch.utils.data import Dataset
import torch
import numpy as np


class AudioWindowDataset(Dataset):
    """Subclass of abstract class Dataset"""

    def __init__(self, samples, labels, ids, window_size=500, step_size=1, how='random-no_step', batch_size=8):
        assert how in ['complete_random', 'local_order',
                       'standard_order', 'total_order']
        self.src = samples
        self.trg = labels
        self.ids = ids
        self.how = how
        self.batch_size = batch_size
        self.window_size = window_size
        self.step_size = int(step_size * window_size)
        self.order = self.create_order()

    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx):
        sample_index = self.order[idx]['index']
        start, end = self.order[idx]['start'], self.order[idx]['end']
        sample = self.src[sample_index][start:end]
        sample = pad_frames(sample, self.window_size, how='0')
        return sample, self.trg[sample_index], self.ids[sample_index]

    def create_order(self):
        return_list = []
        total_order = self.how == 'total_order'
        # order_bool = self.how.split('-')[0] == 'order'
        local_order = self.how == 'local_order'
        complete_random = self.how == 'complete_random'

        step_size = self.step_size if total_order else self.window_size
        for i, x in enumerate(self.src):
            # If sample bigger than window size, the window slides
            if len(x) > self.window_size + step_size:
                n_steps = int(self.window_size/step_size) + \
                    1 if total_order else 1
                start_indexes = [step_size * i for i in range(n_steps)]
                for start_idx in start_indexes:
                    for step_index in range(start_idx, len(x) - self.window_size, self.window_size):
                        return_list.append({
                            'index': i,
                            'start': step_index,
                            'end': step_index + self.window_size
                        })
            else:
                return_list.append({
                    'index': i,
                    'start': 0,
                    'end': self.window_size
                })

        if complete_random:
            np.random.shuffle(return_list)

        else:  # total order or standard order
            # If there is order, the batches are ordered with Nohemien's method
            column_length = len(return_list) // self.batch_size
            indexes = [
                [i + j*column_length for j in range(self.batch_size)] for i in range(column_length)]
            indexes = [x for sublist in indexes for x in sublist]
            return_list = [return_list[i] for i in indexes]

            if local_order:
                list_of_list = [return_list[i:i+self.batch_size]
                                for i in range(0, len(return_list), self.batch_size)]
                np.random.shuffle(list_of_list)
                return_list = [x for sublist in list_of_list for x in sublist]

        return return_list


class Batch:
    def __init__(self, batch, how='class'):
        if how == 'class':
            self.src = torch.stack([torch.FloatTensor(x[0]) for x in batch])
            self.trg = torch.stack([x[1] for x in batch]).long()
            self.ids = [x[2] for x in batch]


def collate_fn(batch):
    return Batch(batch, how='class')


def load(data_path, cv):
    # Loading from cross validation splits
    print('[INFO] Loading data...')
    train_sessions_f = [np.load(
        data_path + '/sess{}_features.npy'.format(i)) for i in range(1, 6) if i != cv]
    train_sessions_l = [np.load(
        data_path + '/sess{}_label.npy'.format(i)) for i in range(1, 6) if i != cv]

    train_features = [x for sublist in train_sessions_f for x in sublist]
    train_labels = [x for sublist in train_sessions_l for x in sublist]

    test_features = np.load(data_path + '/sess{}_features.npy'.format(cv))
    test_labels = np.load(data_path + '/sess{}_label.npy'.format(cv))

    # Labels are stored for different purposes, here we are interested only in
    # classification so we take the first element
    train_labels = [x[0] for x in train_labels]
    test_labels = [x[0] for x in test_labels]

    # Normalizing over the neutral features
    print('[INFO] Normalizing features...')
    neutral_feat = [sublist for i, x in enumerate(
        train_labels) if x == 'neu' for sublist in train_features[i]]
    mean_feat = np.mean(neutral_feat, axis=0)
    std_feat = np.std(neutral_feat, axis=0)
    train_features = [[(sublist - mean_feat) / (std_feat + 0.0001)
                       for sublist in x] for x in train_features]
    test_features = [[(sublist - mean_feat) / (std_feat + 0.0001)
                      for sublist in x] for x in test_features]

    # String to category labels
    print('[INFO] Labels to categorical...')
    train_labels, test_labels = to_categorical([train_labels, test_labels])

    # Ids: you probably won't need them
    train_ids = np.arange(len(train_features))
    test_ids = np.arange(len(train_features), len(
        test_features) + len(train_features))

    return train_features, train_labels, train_ids, test_features, test_labels, test_ids, num_class


def to_categorical(label_list):
    """
    Given a label list, ex: [train_labels, val_labels], returns the categorical int values to Tensor.

    :param label_list: list of lists of categorical (strings) labels
    :return: list of lists of categorical (int to Tensor) labels
    :return: number of different labels
    """

    labels_set = sorted(
        list(set([x for sublist in label_list for x in sublist])))
    labels_dict = {k: v for v, k in enumerate(labels_set)}

    print(labels_dict)

    return_list = []
    for list_ in label_list:
        categorical = np.zeros(shape=(len(list_),))
        for i in range(len(list_)):
            categorical[i] = labels_dict[list_[i]]

        return_list.append(torch.FloatTensor(categorical))
    return return_list[0], return_list[1], len(labels_dict)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, criterion, opt=None, weights=None):
        self.criterion = criterion
        self.opt = opt
        self.w = weights

    def __call__(self, x, y):
        loss = self.criterion(x, y)
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss.item()


def pad_frames(signal, n_frames, how='replicate'):
    """
    Pad a signal to n_frames.

    :param signal: input to pad. Shape is frames X n_features where frames < n_frames
    :param n_frames: number of frames to pad to
    :param how: replicate the beginning of the signal or 0 padding
    :return: padded signal
    """
    n_features = len(signal[0])
    if how == '0':
        return signal + (n_frames - len(signal)) * [n_features*[0]]
    if how == 'replicate':
        while len(signal) != n_frames:
            signal += signal[:(n_frames - len(signal))]
        return signal
