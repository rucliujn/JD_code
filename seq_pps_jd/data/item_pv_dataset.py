import torch
from torch.utils.data import Dataset
import numpy as np
import random
import others.util as util

from collections import defaultdict

""" load training, validation and test data
u,Q,i for a purchase i given u, Q
        negative samples i- for u, Q
read reviews of u, Q before i (previous m reviews)
reviews of others before r_i (review of i from u)
load test data
for each review, collect random words in reviews or words in a sliding window in reviews.
                 or all the words in the review
"""

class ItemPVDataset(Dataset):
    def __init__(self, args, global_data, prod_data):
        self.args = args
        self.prod_pad_idx = global_data.product_size
        self.word_pad_idx = global_data.vocab_size - 1
        self.train_review_only = args.train_review_only
        self.uprev_review_limit = args.uprev_review_limit
        self.global_data = global_data
        self.prod_data = prod_data
        if prod_data.set_name == "train":
            self._data = self.collect_train_samples(self.global_data, self.prod_data)
        else:
            self._data = self.collect_test_samples(self.global_data, self.prod_data, args.candi_batch_size)

    def collect_test_samples(self, global_data, prod_data, candi_batch_size=1000):
        #Q, review of u + review of pos i, review of u + review of neg i;
        #words of pos reviews; words of neg reviews, all if encoder is not pv
        test_data = []
        test_len = len(prod_data.test_lineno_pid)
        for i in range(test_len) :
            test_data.append(prod_data.test_query[i][1],prod_data.test_lineno_pid[i][0],prod_data.test_lineno_pid[i][1:],prod_data.test_target[i][1])
            #query, linno, candi_set, target
            #test_lineno_pid is a list
        return test_data


    def collect_train_samples(self, global_data, prod_data):
        #Q, review of u + review of pos i, review of u + review of neg i;
        #words of pos reviews; words of neg reviews, all if encoder is not pv
        train_data = []
        for lineno, loc in self.prod_data.train_lineno_loc :
            train_data.append([lineno, loc])
        return train_data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]
