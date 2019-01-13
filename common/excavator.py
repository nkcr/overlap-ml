from common import data
from common.utils import get_logger
from torch.autograd import Variable
import numpy as np
import hashlib
import os
import torch

"""Contains the DataSelector class, which is used by the models.

Author: Noémien Kocher
Date: Fall 2018
Unit test: excavator_test.py
"""


class DataSelector:
    """This class is responsible for all the transactions with the
    datasets. It is espacially used as an attempt to perform better
    datas election on the training set.
    """

    def __init__(self, args):
        self.args = args
        self.logger = get_logger(self.args)

        data_hash = (args.data + args.main_model).encode()
        fn = 'corpus.{}.data'.format(
            hashlib.md5(data_hash).hexdigest())
        if os.path.exists(fn):
            self.logger.info('(excavator) Loading cached dataset...')
            corpus = torch.load(fn)
        else:
            self.logger.info('(excavator) Producing dataset...')
            corpus = data.Corpus(args.data)
            torch.save(corpus, fn)

        self.ntokens = len(corpus.dictionary)

        self.train_data = self.batchify(corpus.train, 1)
        self.val_data = self.batchify(corpus.valid, self.args.eval_batch_size)
        self.test_data = self.batchify(corpus.test, self.args.test_batch_size)

        # train_data stats
        self.nitems = self.train_data.size(0) // self.args.bptt
        # This ensures that the +1 for the target won't make out of bound
        self.nitems = self.nitems - 1 if self.nitems * \
            self.args.bptt == self.train_data.size(0) else self.nitems
        self.logger.info(f"(excavator) Number of items: {self.nitems}")

        # batch id to data id
        # here we forget the last chunk of data not multiple of bptt
        self.b2d = self.init_b2d()

        # this list holds the idx of datapoints for each batch
        self._current_seq = self.manual_seq(args.batch_size)

        # keeps track of the number of batches
        self.nbatch = len(self.current_seq)

    @property
    def data_size(self):
        return len(self.current_seq) * len(self.current_seq[0])

    @property
    def batch_size(self):
        return len(self.current_seq[0])

    @property
    def current_seq(self):
        return self._current_seq

    @current_seq.setter
    def current_seq(self, value):
        self._current_seq = value
        self.nbatch = len(self.current_seq)
        self.logger.info(f"(excavator) New current_seq. "
                         f"Shape: {np.shape(self.current_seq)}")

    # ________________________________________________________________________
    # Utility / Initializations

    def init_b2d(self):
        return [i*self.args.bptt for i in range(self.nitems)]

    def init_b2d_overlap(self, overlap):
        assert self.args.bptt % overlap == 0, "overlap must divide bptt"
        shift = self.args.bptt // overlap
        # +1 is for the target token
        return [i*shift for i in range(self.nitems+1)]

    def batchify(self, data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous().to(self.args.device)
        self.logger.info(f"(utils) Data size: {data.size()}.")
        return data

    def get_batch(self, source, i, seq_len=None):
        """Legacy method"""
        seq_len = min(seq_len if seq_len else self.args.bptt,
                      len(source) - 1 - i)
        data = Variable(source[i:i+seq_len])
        # target = Variable(source[i+1:i+1+seq_len].view(-1))
        target = Variable(source[i+1:i+1+seq_len])
        return data, target

    def get_or_create_rstate(self):
        """rstate is Random State, which is used to have a separate random
        state for shuffling
        """
        if not hasattr(self, "rstate"):
            self.rstate = np.random.RandomState(self.args.seed_shuffle)
        return self.rstate

    # ________________________________________________________________________
    # Train seq iterator

    def train_seq(self):
        """Iterator over the train batches using `current_seq`

        `current_seq` contains a 2D list of datapoint ids, where each row is a
        batch. This method iterates over the row (ie. batch) and yields the
        tokens corresponding to the datapoints ids. It yields the data along
        with its target.
        To use this method we must assume that `train_data` is of shape (-1, 1)
        ie. batch_size==1.

        Shape:
            data: (bptt, batch size)
            target: (bptt, batch size)
        """
        assert self.train_data.size(1) == 1, "train_data must be one-column"
        bptt = self.args.bptt
        for batches_id in self.current_seq:
            data = self.train_data[[i for k in batches_id for i in np.arange(
                self.b2d[k], self.b2d[k] + bptt)]]
            target = self.train_data[[i for k in batches_id for i in np.arange(
                self.b2d[k]+1, self.b2d[k]+1 + bptt)]]
            yield data.view(-1, bptt).t().contiguous(), \
                target.view(-1, bptt).t().contiguous()

    # ________________________________________________________________________
    # Train seq initialization

    def manual_seq(self, bsize):
        """Returns a `current_seq` for a standard linear data selection

        The current_seq is a list that indicates the datapoints idx of each
        batch, where each row is a batch.

        For example, if bsize equals 4 and there is 14 datapoints, here is the
        corresponding current_seq:

        0  3  6  9
        1  4  7  10
        2  5  8  11
        """
        nbatch = int(self.nitems // bsize)
        item_idx = np.arange(0, nbatch*bsize)
        item_idx = item_idx.reshape(bsize, nbatch).T
        return item_idx.tolist()

    def overlap_c_seq(self, bsize, overlap):
        """Variant of the overlap sequence, with contiguous sequence.

        To understand how the overlapping works, lets suppose our dataset
        contains 18 tokens and `bptt` is set to 6.

        - With no overlapping, we would build 3 datapoints:

        |-----0-----|-----1-----|-----2-----|
         a b c d e f g h i j k l m n o p q r

        - With an overlapping of 2, we end up with 5 datapoints:

            |-----1-----|-----3-----|
        |-----0-----|-----2-----|-----4-----|
         a b c d e f g h i j k l m n o p q r


        - With an overlapping of 3, we end up with 7 datapoints:

                |-----2-----|-----5-----|
            |-----1-----|-----4-----|
        |-----0-----|-----3-----|-----6-----|
         a b c d e f g h i j k l m n o p q r

        In the naive form, we would build the datapoints sequence in the
        natural order: 0, 1, 2, 3, 4, 5, 6.
        With the contiguous variant, we append each sub sequence, wich yields
        the following order: 0, 3, 6, 1, 4, 2, 5.

        With the overlapping of 3, the sub-sequences are gradually shifted by
        two, because 6 / 3 = 2. With the overlapping of 2, sub-sequences are
        shifted by 3, because 6 / 2 = 3. Hence, the number of tokens per
        datapoints must be divisible by the overlapping.

        With the overlapping of 3 and a batch size of 2, here is the
        corresponding current_seq in the naive and contiguous form:

        contiguous      naive
           0  1          0  3
           3  4          1  4
           6  2          2  5
        """
        dsize = self.train_data.size(0)
        shift = self.args.bptt // overlap
        # ndatapoints = sum([(dsize-i*shift) // self.args.bptt
        #                    for i in range(overlap)])
        # result = []
        # for i in range(overlap):
        #     for j in range(ndatapoints//overlap):
        #         result.append(i+j*overlap)
        # item_idx = np.array(result[:len(result)-len(result) % bsize])
        # nbatch = item_idx.size // bsize
        # item_idx = item_idx.reshape(bsize, nbatch).T
        # self.nitems = ndatapoints
        # self.b2d = self.init_b2d_overlap(overlap)

        len_sub_seq = dsize // self.args.bptt
        dp_seq = np.array(range(overlap*len_sub_seq)
                          ).reshape(len_sub_seq, overlap)
        dp_seq = dp_seq.T.reshape(-1)
        dp_seq = dp_seq[(dp_seq*shift+self.args.bptt) < dsize]

        dp_seq = dp_seq[:len(dp_seq)-len(dp_seq) % bsize]
        ndatapoints = dp_seq.size
        nbatch = ndatapoints // bsize
        dp_seq = dp_seq.reshape(bsize, nbatch).T
        self.nitems = ndatapoints
        self.b2d = self.init_b2d_overlap(overlap)
        return dp_seq.tolist()

    # ________________________________________________________________________
    # Train seq update

    def shuffle_row_train_seq(self):
        self.logger.info("(excavator) Shuffe row (row-wise) train_seq")
        rstate = self.get_or_create_rstate()
        rstate.shuffle(self.current_seq)

    def shuffle_col_train_seq(self):
        self.logger.info("(excavator) Shuffle col (column-wise) train_seq")
        rstate = self.get_or_create_rstate()
        temp = np.array(self.current_seq)
        rstate.shuffle(temp.T)
        self.current_seq = temp.tolist()

    def shuffle_each_row_train_seq(self):
        self.logger.info("(excavator) Shuffle each row "
                         "(row individually) train_seq")
        rstate = self.get_or_create_rstate()
        temp = np.array(self.current_seq)
        np.apply_along_axis(rstate.shuffle, 1, temp)
        self.current_seq = temp.tolist()

    def shuffle_full_train_seq(self):
        self.logger.info("(excavator) Shuffle full (row+col wise) train_seq")
        rstate = self.get_or_create_rstate()
        temp = np.array(self.current_seq)
        shape = temp.shape
        temp = temp.reshape(-1)
        rstate.shuffle(temp)
        self.current_seq = temp.reshape(shape).tolist()
