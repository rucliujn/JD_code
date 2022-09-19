import torch
import numpy as np

from others.logging import logger, init_logger
from collections import defaultdict
import others.util as util
import gzip
import os

class ProdSearchData():
    def __init__(self, args, input_train_dir, set_name, global_data):
        self.args = args
        self.neg_per_pos = args.neg_per_pos
        self.set_name = set_name
        self.global_data = global_data
        self.product_size = global_data.product_size
        self.vocab_size = global_data.vocab_size
        self.sub_sampling_rate = None
        self.neg_sample_products = None
        self.word_dists = None
        self.subsampling_rate = args.subsampling_rate
        self.uq_pids = None
        if args.fix_emb:
            self.subsampling_rate = 0

        if set_name == "train":
            self.train_lineno_loc = GlobalProdSearchData.read_arr_from_lines(
                     "{}/{}_train_lineno_loc.txt.gz".format(input_train_dir, set_name))
        if set_name == "test":
            self.test_lineno_pid = GlobalProdSearchData.read_arr_from_lines(
                     "{}/{}_test_lineno_pid.txt.gz".format(input_train_dir, set_name))
            self.test_query = GlobalProdSearchData.read_arr_from_lines(
                     "{}/{}_test_query.txt.gz".format(input_train_dir, set_name))
            self.test_target = GlobalProdSearchData.read_arr_from_lines(
                     "{}/{}_test_target.txt.gz".format(input_train_dir, set_name))
        
        self.product_distribute = np.ones(self.product_size)
        self.product_dists = self.neg_distributes(self.product_distribute)
            

    def read_ranklist(self, fname, product_asin2ids):
        uq_pids = defaultdict(list)
        with open(fname, 'r') as fin:
            for line in fin:
                arr = line.strip().split(' ')
                uid, qid = arr[0].split('_')
                asin = arr[2]
                uq_pids[(uid, int(qid))].append(product_asin2ids[asin])
        return uq_pids


    def initialize_epoch(self):
        return


    def neg_distributes(self, weights, distortion = 0.75):
        #print weights
        weights = np.asarray(weights)
        #print weights.sum()
        wf = weights / weights.sum()
        wf = np.power(wf, distortion)
        wf = wf / wf.sum()
        return wf


class GlobalProdSearchData():
    def __init__(self, args, data_path, input_train_dir):

        self.product_ids = self.read_lines("{}/product.txt.gz".format(data_path))
        self.product_asin2ids = {x:i for i,x in enumerate(self.product_ids)}
        self.product_size = len(self.product_ids)


        self.words = self.read_lines("{}/vocab.txt.gz".format(data_path))
        self.vocab_size = len(self.words) + 1
        self.word_pad_idx = self.vocab_size-1

        
        product_dist = self.read_arr_from_lines("{}/neighbour_product.txt.gz".format(data_path))
        self.product_dist = {str(line[0]) + '_' + str(line[1]) : line[2] for line in product_dist}

        self.his_prod_list, self.his_query_list = self.read_his_from_lines("{}/user_history.txt.gz".format(data_path))
        

    @staticmethod
    def read_arr_from_lines(fname):
        line_arr = []
        with gzip.open(fname, 'rt') as fin:
            for line in fin:
                arr = line.strip().split(' ')
                filter_arr = []
                for idx in arr:
                    if len(idx) < 1:
                        continue
                    filter_arr.append(int(idx))
                line_arr.append(filter_arr)
        return line_arr

    @staticmethod
    def read_lines(fname):
        arr = []
        with gzip.open(fname, 'rt') as fin:
            for line in fin:
                arr.append(line.strip())
        return arr
    
    @staticmethod
    def read_his_from_lines(fname) :
        his_query_list = []
        his_prod_list = []
        with gzip.open(fname, 'rt') as fin :
            for line in fin:
                query_prod_pair_list = line.strip().split('\t')
                for query_prod_pair in query_prod_pair_list :
                    query_prod_pair_num = [int(i) for i in query_prod_pair.strip().split(' ')]
                    his_prod_list.append(query_prod_pair_num[0])
                    his_query_list.append(query_prod_pair_num[1:])
        return his_prod_list, his_query_list
