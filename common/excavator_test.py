import unittest
import mock
from mock import Mock
import numpy as np
from excavator import DataSelector
import torch

"""Simple test class for DataSelector

Author: Noémien Kocher
Date: Fall 2018

Run this class with `python3 excavator_test.py` or
                    `python3 unittest -m excavator_test.py`
"""


class DataSelectorTest(unittest.TestCase):

    def setUp(self):
        self.that = Mock(args=Mock())
        return

    def test_train_seq(self):
        """The train_seq outputs data, target based on the `current_seq`
        """
        self.that.train_data = torch.tensor(np.arange(0, 25)).view(-1, 1)
        self.that.args.bptt = 4
        self.that.b2d = [0, 4, 8, 12, 16, 20]
        self.that.current_seq = [[0, 2], [1, 4]]

        expected_data = []
        expected_target = []
        expected_data.append([
            [0, 8],
            [1, 9],
            [2, 10],
            [3, 11]
        ])
        expected_target.append([1, 9, 2, 10, 3, 11, 4, 12])
        expected_data.append([
            [4, 16],
            [5, 17],
            [6, 18],
            [7, 19]
        ])
        expected_target.append([5, 17, 6, 18, 7, 19, 8, 20])
        seq = DataSelector.train_seq(self.that)
        for i, (data, target) in enumerate(seq):
            self.assertEqual(
                data.numpy().tolist(), expected_data[i])
            self.assertEqual(
                target.numpy().tolist(), expected_target[i])

    def test_manual_seq(self):
        """We threat our dataset as N blocks composed of self.bptt tokens.
        Each block has an idx relative to the number of block. We use b2d
        to convert from the block id to the absolute dataset tokens id.

        Here we have our blocks:

        b0 b1 b2 b3 b4 b5

        That we divide to form batches:

        b0 b2 b4 <--- first batch
        b1 b3 b5

        b0 will contains the first bptt=4 tokens of the train dataset
        b3 will contains tokens 12 to 15 (3*4)
        """
        self.that.train_data = torch.tensor(np.arange(0, 25)).view(-1, 1)
        self.that.args.bptt = 4
        self.that.nitems = 6
        self.that.b2d = [0, 4, 8, 12, 16, 20]
        self.that.current_seq = [[0, 2], [1, 4]]

        expected = [[0, 2, 4], [1, 3, 5]]
        seq = DataSelector.manual_seq(self.that, 3)
        self.assertEqual(seq, expected)

        # Second test, with more data
        self.that.train_data = torch.tensor(np.arange(0, 40)).view(-1, 1)
        self.that.nitems = 10
        expected = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
        seq = DataSelector.manual_seq(self.that, 3)
        self.assertEqual(seq, expected)

    def test_manuel_train_seq(self):
        self.that.train_data = torch.tensor(np.arange(0, 25)).view(-1, 1)
        self.that.args.bptt = 4
        self.that.nitems = 6
        self.that.b2d = [0, 4, 8, 12, 16, 20]
        self.that.current_seq = DataSelector.manual_seq(self.that, 3)
        expected_data = []
        expected_target = []
        expected_data.append([
            [0, 8, 16],
            [1, 9, 17],
            [2, 10, 18],
            [3, 11, 19]
        ])
        expected_target.append([
            1, 9, 17, 2, 10, 18, 3, 11, 19, 4, 12, 20
        ])
        expected_data.append([
            [4, 12, 20],
            [5, 13, 21],
            [6, 14, 22],
            [7, 15, 23]
        ])
        expected_target.append([
            5, 13, 21, 6, 14, 22, 7, 15, 23, 8, 16, 24
        ])
        seq = DataSelector.train_seq(self.that)
        for i, (data, target) in enumerate(seq):
            self.assertEqual(
                data.numpy().tolist(), expected_data[i])
            self.assertEqual(
                target.numpy().tolist(), expected_target[i])

    def test_overlap_c_seq(self):
        train_data = torch.tensor(np.arange(0, 25)).view(-1, 1)
        batch_size = 6
        bptt = 6
        overlap = 3
        # max_end should be 25
        # data points are [0,1,2,...19]
        # number of data points is 4+3+3 = 10
        # number of batches is 10 // 6 = 1
        expected = [
            [0, 3, 6, 1, 4, 7]
        ]
        self.that.train_data = train_data
        self.that.args.bptt = bptt
        result = DataSelector.overlap_c_seq(self.that, batch_size, overlap)
        self.assertEqual(result, expected)
        self.assertEqual(self.that.nitems, 10)

        batch_size = 4
        bptt = 4
        overlap = 2
        # number of data points is 6+5 = 11
        # number of batches is 11 // 4 = 2
        expected = [
            [0, 4, 8, 3],
            [2, 6, 1, 5]
        ]
        self.that.args.bptt = bptt
        result = DataSelector.overlap_c_seq(self.that, batch_size, overlap)
        self.assertEqual(result, expected)
        self.assertEqual(self.that.nitems, 11)

        batch_size = 4
        bptt = 4
        overlap = 4
        # number of data points is 6+5+5+6 = 22
        # number of batches is 21 // 4 = 5
        expected = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19]
        ]
        self.that.args.bptt = bptt
        result = DataSelector.overlap_c_seq(self.that, batch_size, overlap)
        self.assertEqual(result, expected)
        self.assertEqual(self.that.nitems, 22)


if __name__ == "__main__":
    unittest.main()
