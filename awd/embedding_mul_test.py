import unittest
import mock
from mock import Mock
import numpy as np
from embedding_mul import EmbeddingMul
import torch
import argparse
import logging

"""Simple test class for EmbeddingMul class

Author: Noémien Kocher
Date: Fall 2018

Run this class with `python3 embedding_mul_test.py` or
                    `python3 unittest -m embedding_mul_test.py`
"""


class EmbeddingMulTest(unittest.TestCase):

    def test_forward(self):
        """Simple test using tensors"""
        emsize = 3
        num_token = 4
        emm = EmbeddingMul(num_token, 'cpu')

        #  0   1   2
        #  3   4   5
        #  6   7   8
        #  9  10  11
        weights = torch.tensor(range(emsize*num_token)
                               ).view(num_token, emsize).float()

        input = torch.tensor(
            [[3.0, 2, 1], [0, 3, 2], [2, 2, 0]])
        expected = torch.tensor([
            [[9, 10, 11], [6, 7, 8], [3, 4, 5]],
            [[0, 1, 2], [9, 10, 11], [6, 7, 8]],
            [[6, 7, 8], [6, 7, 8], [0, 1, 2]]
        ]).float()

        result = emm(input, weights, -1)

        self.assertEqual(result.detach().numpy().tolist(),
                         expected.numpy().tolist())

    def test_forward2(self):
        """Test using the original embedding module from pytorch"""
        emsize = 3
        num_token = 4
        emm = EmbeddingMul(num_token, 'cpu')

        #  0   1   2
        #  3   4   5
        #  6   7   8
        #  9  10  11
        weights = torch.tensor(range(emsize*num_token)
                               ).view(num_token, emsize).float()

        true_em = torch.nn.Embedding(num_token, emsize)
        true_em.weight.data = weights

        input = torch.tensor([[3.0, 2, 1], [0, 3, 2], [2, 2, 0]])
        expected = true_em(input.long()).float()

        result = emm(input, weights, -1)

        self.assertEqual(result.detach().numpy().tolist(),
                         expected.detach().numpy().tolist())

    def test_forward_grad(self):
        num_tokens = 3
        input = torch.tensor([[0, 1]])
        weights = torch.tensor([[1.0, 2], [3, 4], [5, 6]])
        emm = EmbeddingMul(num_tokens, 'cpu')
        emm._requires_grad = True
        X_hat = emm(input, weights, -1)
        W = torch.tensor([[2.0], [3]])
        X_hhat = torch.mm(X_hat[0], W)
        y = torch.sum(X_hhat)
        y.backward()

        # Those results were computed by hand
        expected = torch.tensor([
            [[8.0, 18.0, 28.0], [8.0, 18.0, 28.0]]
        ])
        result = emm.last_oh.grad
        self.assertEqual(result.numpy().tolist(), expected.numpy().tolist())


if __name__ == "__main__":
    unittest.main()
