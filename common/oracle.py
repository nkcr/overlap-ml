import os
from common.utils import get_logger
import numpy as np
import json
from collections import defaultdict

"""Holds the StatsKeeper class

Author: Noémien Kocher
Date: Fall 2018
"""


class StatsKeeper():
    """This class is reponsible of saving the stats about batches, taking
    into account the epoch. It relies on its attributes' states."""

    def __init__(self, args, path):
        self.args = args
        if os.path.isabs(path):
            model_id = os.path.basename(os.path.normpath(args.model_dir))
            self.path = os.path.join(path, model_id)
        else:
            self.path = os.path.join(args.model_dir, path)
        self.logger = get_logger(args)

        # epoch
        self.epoch = -1
        self.current_data = None
        self.past_data = None
        self._init_new_epoch(1)

        # prior
        self.prior_epoch = -1
        self.prior_scores = None
        self._init_new_prior_epoch(1)

        # init
        self._init_folder(self.path)

        # General purpose data
        self.data = defaultdict(lambda: [])
        self.data_epoch = defaultdict(lambda: -1)

    def end(self):
        self._save_epoch()
        self.save_prior_epoch()

    #######
    # Epoch
    #######
    def add_sample(self, epoch, inner_step, value):
        if epoch != self.epoch:
            self._save_epoch()
            self._init_new_epoch(epoch)
        self.current_data.append(value)

    def _save_epoch(self):
        data_path = os.path.join(self.path, f"epoch_{self.epoch}.json")
        # np.save(data_path, np.array(self.current_data))
        with open(data_path, 'w') as outfile:
            json.dump(self.current_data, outfile)

    def _init_new_epoch(self, epoch):
        self.logger.info(f"(oracle) Init new epoch: {epoch}")
        self.epoch = epoch
        self.past_data = self.current_data
        self.current_data = []

    #############
    # Prior epoch
    #############
    def add_prior_sample(self, epoch, value):
        if epoch != self.prior_epoch:
            self._init_new_prior_epoch(epoch)
        self.prior_scores.append(value)

    def save_prior_epoch(self):
        data_path = os.path.join(
            self.path, f"epoch_{self.prior_epoch}_prior.json")
        # np.save(data_path, np.array(self.current_data))
        with open(data_path, 'w') as outfile:
            json.dump(self.prior_scores, outfile)

    def _init_new_prior_epoch(self, epoch):
        self.logger.info(f"(oracle) Init new prior epoch: {epoch}")
        self.prior_epoch = epoch
        self.prior_scores = []

    def _init_folder(self, path):
        if not os.path.exists(path):
            self.logger.info(f"(oracle) Creates {path}")
            os.makedirs(path)

    def save_seq(self, seq):
        """Saves the sequence as JSON, referenced by the epoch"""
        if type(seq) is np.ndarray:
            seq = seq.tolist()
        data_path = os.path.join(self.path, f"epoch_{self.epoch}_seq.json")
        # np.save(data_path, np.array(seq))
        with open(data_path, 'w') as outfile:
            json.dump(seq, outfile)

    #############################
    # General purpose stats saver
    #############################
    def add_data(self, name, epoch, inner_step, value):
        if self.data_epoch[name] == -1:
            self._init_new_data_epoch(name, epoch)
        if epoch != self.data_epoch[name]:
            self._save_data_epoch(name)
            self._init_new_data_epoch(name, epoch)
        self.data[name].append(value)

    def _save_data_epoch(self, name):
        data_path = os.path.join(
            self.path, f"epoch_{self.data_epoch[name]}_{name}")
        np.save(data_path, np.array(self.data[name]))
        # with open(data_path, 'w') as outfile:
        #     json.dump(self.data[name], outfile)

    def _init_new_data_epoch(self, name, epoch):
        self.logger.info(f"(oracle) Init new epoch: {epoch} (for '{name}')")
        self.data_epoch[name] = epoch
        self.data[name] = []
