from re import U
import torch
from torch.utils.data import DataLoader
import others.util as util
import numpy as np
import random
from data.batch_data import ItemPVBatch


class ItemPVDataloader(DataLoader):
    def __init__(self, args, dataset, prepare_pv=True, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None):
        super(ItemPVDataloader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            batch_sampler=batch_sampler, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
            worker_init_fn=worker_init_fn, collate_fn=self._collate_fn)
        self.args = args
        self.prod_pad_idx = self.dataset.prod_pad_idx
        self.word_pad_idx = self.dataset.word_pad_idx
        self.global_data = self.dataset.global_data
        self.prod_data = self.dataset.prod_data

    def _collate_fn(self, batch):
        if self.prod_data.set_name == 'train':
            return self.get_train_batch(batch)
        if self.prod_data.set_name == 'test':
            return self.get_test_batch(batch)

    def get_test_batch(self, batch):
        candi_prod_idxs = [entry[2] for entry in batch]
        candi_u_item_idxs = []
        candi_u_item_query_word_idxs = []
        candi_u_item_dist = []
        candi_prod_dist = []
        cnt = 0

        #query, linno, candi_set

        query_word_idxs = [entry[0] for entry in batch]
        query_idxs = [-1 for entry in batch]
        user_idxs = [entry[1] for entry in batch]
        target_prod_idxs = [entry[3] for entry in batch]

        for query_idx, lineno, candi_set in batch :
            u_item_idxs = self.global_data.his_prod_list[lineno].reverse()[:self.args.uprev_review_limit]
            u_item_query_word_idxs = self.global_data.his_query_list[lineno].reverse()[:self.args.uprev_review_limit]

            u_item_dist = [[self.args.dist_pad for i in range(len(u_item_idxs) + 1)]]
            for idx_i in u_item_idxs :
                tmp_list = [self.args.dist_pad]
                for idx_j in u_item_idxs :
                    if (idx_i == idx_j) :
                        tmp_list.append(0)
                        continue
                    pair = str(min(idx_i,idx_j)) + '_' + str(max(idx_i,idx_j))
                    if (pair in self.global_data.product_dist) :
                        tmp_list.append(self.global_data.product_dist[pair])
                    else :
                        tmp_list.append(self.args.dist_pad)
                u_item_dist.append(tmp_list)
            prod_dist = []
            for candi_prod_idx in candi_prod_idxs[cnt] :
                tmp_list = [self.args.dist_pad]
                for idx_i in u_item_idxs :
                    pair = str(min(candi_prod_idx,idx_i)) + '_' + str(max(candi_prod_idx,idx_i))
                    if (pair in self.global_data.product_dist) :
                        tmp_list.append(self.global_data.product_dist[pair])
                    else :
                        tmp_list.append(self.args.dist_pad)
                prod_dist.append(tmp_list)
            
            candi_prod_dist.append(prod_dist)
            candi_u_item_idxs.append(u_item_idxs)
            candi_u_item_query_word_idxs.append(u_item_query_word_idxs)
            candi_u_item_dist.append(u_item_dist)



        candi_prod_idxs = util.pad(candi_prod_idxs, pad_id = self.prod_pad_idx, width = self.args.candi_batch_size)
        candi_u_item_idxs = util.pad(candi_u_item_idxs, pad_id = self.prod_pad_idx)
        candi_u_item_query_word_idxs = util.pad_3d(candi_u_item_query_word_idxs, pad_id = self.global_data.word_pad_idx, width = self.args.uprev_review_limit)
        candi_u_item_dist = util.pad_3d(candi_u_item_dist, pad_id = self.args.dist_pad, dim = 2)
        candi_u_item_dist = util.pad_3d(candi_u_item_dist, pad_id = self.args.dist_pad, dim = 1)
        candi_prod_dist = util.pad_3d(candi_prod_dist, pad_id = self.args.dist_pad, dim = 2)
        candi_prod_dist = util.pad_3d(candi_prod_dist, pad_id = self.args.dist_pad, dim = 1, width = self.args.candi_batch_size)       
        batch = ItemPVBatch(query_word_idxs, target_prod_idxs, candi_u_item_idxs, candi_u_item_query_word_idxs, candi_u_item_dist, 
                            query_idxs=query_idxs, user_idxs=user_idxs, candi_prod_idxs=candi_prod_idxs, candi_prod_dist = candi_prod_dist)
        return batch


    def get_train_batch(self, batch):
        batch_query_word_idxs = []
        batch_u_item_idxs, batch_target_prod_idxs, batch_neg_prod_idxs = [],[],[]
        batch_u_item_query_word_idxs = []
        batch_u_item_dist = []
        batch_target_prod_dist = []
        batch_neg_prod_dist = []
        prod_dists = torch.ones(self.global_data.product_size)

        batch_query_idxs = [entry[1] for entry in batch]
        batch_user_idxs = [entry[0] for entry in batch]


        for lineno, loc in batch :
            batch_query_word_idxs.append(self.global_data.his_query_list[lineno][loc])
            batch_target_prod_idxs.append(self.global_data.his_prod_list[lineno][loc])
            neg_prod_idxs = torch.multinomial(prod_dists, self.args.neg_per_pos, replacement=True).tolist()
            batch_neg_prod_idxs.append(neg_prod_idxs)
            batch_u_item_idxs.append(self.global_data.his_prod_list[lineno][:loc])
            batch_u_item_query_word_idxs.append(self.global_data.his_query_list[lineno][:loc])

            u_item_idxs = self.global_data.his_prod_list[lineno][:loc]
            u_item_query_word_idxs = self.global_data.his_query_list[lineno][:loc]
            prod_idx = self.global_data.his_prod_list[lineno][loc]

            u_item_dist = [[self.args.dist_pad for i in range(len(u_item_idxs) + 1)]]
            target_prod_dist = [self.args.dist_pad]
            neg_prod_dist = [[self.args.dist_pad] for i in range(self.args.neg_per_pos)]
            for idx_i in u_item_idxs :
                tmp_list = [self.args.dist_pad]
                for idx_j in u_item_idxs :
                    if (idx_i == idx_j) :
                        tmp_list.append(0)
                        continue
                    pair = str(min(idx_i,idx_j)) + '_' + str(max(idx_i,idx_j))
                    if (pair in self.global_data.product_dist) :
                        tmp_list.append(self.global_data.product_dist[pair])
                    else :
                        tmp_list.append(self.args.dist_pad)
                u_item_dist.append(tmp_list)

                pos_pair = str(min(prod_idx, idx_i)) + '_' + str(max(prod_idx, idx_i))
                if (pos_pair in self.global_data.product_dist) :
                    target_prod_dist.append(self.global_data.product_dist[pos_pair])
                else :
                    target_prod_dist.append(self.args.dist_pad)
                
                for i, neg_idx in enumerate(neg_prod_idxs) :
                    neg_pair = str(min(neg_idx, idx_i)) + '_' + str(max(neg_idx, idx_i))
                    if (neg_pair in self.global_data.product_dist) :
                        neg_prod_dist[i].append(self.global_data.product_dist[neg_pair])
                    else :
                        neg_prod_dist[i].append(self.args.dist_pad)

        batch_u_item_idxs = util.pad(batch_u_item_idxs, pad_id = self.prod_pad_idx)
        batch_u_item_query_word_idxs = util.pad_3d(batch_u_item_query_word_idxs, pad_id = self.global_data.word_pad_idx, width = self.args.uprev_review_limit)
        batch_u_item_dist = util.pad_3d(batch_u_item_dist, pad_id = self.args.dist_pad, dim = 2)
        batch_u_item_dist = util.pad_3d(batch_u_item_dist, pad_id = self.args.dist_pad, dim = 1)

        batch_target_prod_dist = util.pad(batch_target_prod_dist, pad_id = self.args.dist_pad)
        batch_neg_prod_dist = util.pad_3d(batch_neg_prod_dist, pad_id = self.args.dist_pad, dim = 2)

        batch = ItemPVBatch(batch_query_word_idxs, batch_target_prod_idxs, batch_u_item_idxs, batch_u_item_query_word_idxs, batch_u_item_dist, query_idxs = batch_query_idxs, user_idxs = batch_user_idxs,
                            neg_prod_idxs = batch_neg_prod_idxs, target_prod_dist = batch_target_prod_dist, neg_prod_dist = batch_neg_prod_dist)
        return batch
