import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.text_encoder import AVGEncoder, FSEncoder, get_vector_mean
from models.transformer import TransformerEncoder
from models.neural import MultiHeadedAttention
from models.optimizers import Optimizer
from others.logging import logger
from others.util import pad, load_pretrain_embeddings, load_user_item_embeddings

def read_from_file(data_path, size, len) :
    weight = torch.zeros(size, len)
    with open(data_path, 'r') as fr:
        line_cnt = 0
        for line in fr:
            line_cnt += 1
            if (line_cnt <= 2) :
                continue
            line = line.strip().split()
            weight[line_cnt - 3, :] = torch.FloatTensor([float(t) for t in line]) 
    return weight

class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, sign):
        super(NonLinear, self).__init__()
        self.sign = sign
        self.A = nn.Parameter(torch.randn(input_size, output_size))
        self.B = nn.Parameter(torch.FloatTensor(output_size))

    def forward(self, x):
        A = F.softplus(self.A) * self.sign
        #print(A)
        x = torch.mm(x, A) + self.B
        return x


class ItemTransformerRanker(nn.Module):
    def __init__(self, args, device, vocab_size, product_size, vocab_words, word_dists=None):
        super(ItemTransformerRanker, self).__init__()
        self.args = args
        self.device = device
        self.train_review_only = args.train_review_only
        self.embedding_size = args.embedding_size
        self.vocab_words = vocab_words
        self.word_dists = None
        if word_dists is not None:
            self.word_dists = torch.tensor(word_dists, device=device)
        self.prod_dists = torch.ones(product_size, device=device)
        self.prod_pad_idx = product_size
        self.word_pad_idx = vocab_size - 1
        self.emb_dropout = args.dropout
        self.pretrain_emb_dir = None
        if os.path.exists(args.pretrain_emb_dir):
            self.pretrain_emb_dir = args.pretrain_emb_dir
        self.pretrain_up_emb_dir = None
        if os.path.exists(args.pretrain_up_emb_dir):
            self.pretrain_up_emb_dir = args.pretrain_up_emb_dir
        self.dropout_layer = nn.Dropout(p=args.dropout)

        #self.product_emb = nn.Embedding(product_size+1, self.embedding_size, padding_idx=self.prod_pad_idx)
        '''for pretrain'''
        if self.pretrain_emb_dir is not None:
            prod_emb_fname = "product_emb.txt"
            pretrain_prod_emb_path = os.path.join(self.pretrain_emb_dir, prod_emb_fname)
            prod_emb = read_from_file(pretrain_prod_emb_path, product_size+1, self.embedding_size)
            self.product_emb = nn.Embedding(product_size+1, self.embedding_size, padding_idx=self.prod_pad_idx)
            self.product_emb.weight.data.copy_(prod_emb)
        else :
            self.product_emb = nn.Embedding(product_size+1, self.embedding_size, padding_idx=self.prod_pad_idx)
        
        #self.rating_emb = nn.Embedding(6, self.embedding_size, padding_idx = 0)


        self.product_bias = nn.Parameter(torch.zeros(product_size+1), requires_grad=True)
        self.word_bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)

        '''for pretrain'''
        if self.pretrain_emb_dir is not None:
            word_emb_fname = "word_emb.txt" 
            pretrain_word_emb_path = os.path.join(self.pretrain_emb_dir, word_emb_fname)
            word_emb = read_from_file(pretrain_word_emb_path, vocab_size, self.embedding_size)
            self.word_embeddings = nn.Embedding(
                vocab_size, self.embedding_size, padding_idx=self.word_pad_idx)
            self.word_embeddings.weight.data.copy_(word_emb)
        else:
            self.word_embeddings = nn.Embedding(
                vocab_size, self.embedding_size, padding_idx=self.word_pad_idx)
        
        if self.args.model_name == "item_transformer":
            self.transformer_encoder = TransformerEncoder(
                    self.embedding_size, args.ff_size, args.heads,
                    args.dropout, args.inter_layers)

        if args.query_encoder_name == "fs":
            self.query_encoder = FSEncoder(self.embedding_size, self.emb_dropout)
        else:
            self.query_encoder = AVGEncoder(self.embedding_size, self.emb_dropout)
        
        self.F_matrix = NonLinear(input_size = 1, output_size = args.heads, sign = -1.0)
        #self.G_matrix = NonLinear(input_size = 1, output_size = args.heads, sign = 1.0)
        self.feature_layer = nn.Sequential(nn.Linear(self.args.uprev_review_limit + 1, 1), nn.Tanh())
        self.score_layer = nn.Linear(2, 1)
        #self.match_score_layer = nn.Linear(2, 1)
        self.match_lambda = nn.Parameter(torch.ones(1) * 0.5)
        self.dist_layer = nn.Linear(3 * self.embedding_size, 1)

        self.decoder = nn.GRUCell(self.embedding_size, self.embedding_size)
        self.decoder_mlp = nn.Linear(self.embedding_size, self.embedding_size)
        self.decoder_loss = nn.CrossEntropyLoss(ignore_index = -1)


        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')#by default it's mean

        self.initialize_parameters(logger) #logger
        self.to(device) #change model in place
        self.item_loss = 0
        self.ps_loss = 0
        self.dist_loss = 0
        self.decode_loss = 0
        self.alpha = self.args.alpha
        self.beta = self.args.beta
        


    def clear_loss(self):
        self.item_loss = 0
        self.ps_loss = 0
        self.dist_loss = 0
        self.decode_loss = 0

    def load_cp(self, pt, strict=True):
        self.load_state_dict(pt['model'], strict=strict)

    def test(self, batch_data):
        return self.test_dotproduct(batch_data)

    def test_dotproduct(self, batch_data):
        query_word_idxs = batch_data.query_word_idxs
        u_item_idxs = batch_data.u_item_idxs
        u_item_query_word_idxs = batch_data.u_item_query_word_idxs#batch, prev_item_count, max_query_len
        u_item_dist = batch_data.u_item_dist#batch, prev_item_count + 1, prev_item_count + 1
        candi_prod_dist = batch_data.candi_prod_dist#batch, candi_k, prev_item_count + 1
        candi_prod_idxs = batch_data.candi_prod_idxs
        batch_size, prev_item_count = u_item_idxs.size()
        _, candi_k = candi_prod_idxs.size()

        
        query_word_emb = self.word_embeddings(query_word_idxs)
        query_org_emb = self.query_encoder(query_word_emb, query_word_idxs.ne(self.word_pad_idx))
        query_emb = query_org_emb.unsqueeze(1)

        u_item_query_word_idxs = u_item_query_word_idxs.view(batch_size * prev_item_count, -1)
        u_item_query_word_emb = self.word_embeddings(u_item_query_word_idxs)#batch * prev_item_count, max_query_len, emb
        u_item_query_emb = self.query_encoder(u_item_query_word_emb, u_item_query_word_idxs.ne(self.word_pad_idx)).view(batch_size, prev_item_count, self.embedding_size)#batch, prev_item_count, emb
        
        u_item_query_emb = torch.cat([query_emb, u_item_query_emb], dim=1)#batch, (prev_item_count+1), emb
        u_item_sim_matrix = torch.cosine_similarity(u_item_query_emb.unsqueeze(1), u_item_query_emb.unsqueeze(2), dim=-1).view(-1, prev_item_count+1, prev_item_count+1)#batch, prev_item_count+1, prev_item_count+1

        u_item_sim_matrix_G = u_item_sim_matrix.unsqueeze(1).expand(-1, self.args.heads, -1, -1)
        
        u_item_sim_matrix_F = self.F_matrix(u_item_dist.view(-1, 1)).view(-1, prev_item_count+1, prev_item_count+1, self.args.heads).permute(0, 3, 1, 2)
        u_item_sim_matrix_F[:,:,0,:] = 0.0
        u_item_sim_matrix_F[:,:,:,0] = 0.0
        u_item_sim_matrix = u_item_sim_matrix_G + u_item_sim_matrix_F #
   
        column_mask = torch.ones(batch_size, 1, dtype=torch.uint8, device=query_word_idxs.device)
        u_item_mask = u_item_idxs.ne(self.prod_pad_idx)
        u_item_mask = u_item_mask
        column_mask = column_mask
        candi_item_seq_mask = torch.cat([column_mask, u_item_mask], dim=1)
        candi_item_emb = self.product_emb(candi_prod_idxs) #batch_size, candi_k, embedding_size

        u_item_emb = self.product_emb(u_item_idxs)
        cls_emb = torch.zeros(batch_size, 1, self.embedding_size, dtype=torch.float, device=query_word_idxs.device)
        candi_sequence_emb = torch.cat([cls_emb,u_item_emb], dim=1)

        out_pos = -1 if self.args.use_item_pos else 0
        top_vecs = self.transformer_encoder.encode(candi_sequence_emb,candi_item_seq_mask, u_item_sim_matrix, use_pos=self.args.use_pos_emb)
        candi_out_emb = top_vecs[:,out_pos,:]#batch_size, embedding_size
        item_out_emb = top_vecs[:,1:,:]#batch, prev_item_count, embedding_size
        
        #dist_loss = self.graph_dist_loss(u_item_dist, item_out_emb, u_item_mask)
        #decode_loss = self.seq_decode_loss(candi_out_emb, u_item_emb, cls_emb, u_item_mask)           
        
        candi_out_emb = candi_out_emb.repeat(1, candi_k).view(-1, self.embedding_size)#batch_size*candi_k, embedding_size
        candi_query_emb = query_org_emb.repeat(1, candi_k).view(-1, self.embedding_size)#batch_size*candi_k, embedding_size
        
        candi_u_match_scores = torch.bmm(candi_out_emb.unsqueeze(1), candi_item_emb.view(batch_size*candi_k, -1).unsqueeze(2)).view(-1, 1)
        candi_q_match_scores = torch.bmm(candi_query_emb.unsqueeze(1), candi_item_emb.view(batch_size*candi_k, -1).unsqueeze(2)).view(-1, 1)
        candi_match_scores = self.match_lambda * candi_u_match_scores + (1 - self.match_lambda) * candi_q_match_scores

        candi_feature_scores = self.feature_layer(candi_prod_dist.view(-1, prev_item_count + 1)).view(-1, 1)
        candi_scores = self.score_layer(torch.cat([candi_feature_scores, candi_match_scores], dim=1)).view(batch_size, candi_k)

        if self.args.sim_func == "bias_product":
            candi_bias = self.product_bias[candi_prod_idxs.view(-1)].view(batch_size, candi_k)
            candi_scores += candi_bias
        return candi_scores


    def forward(self, batch_data, train_pv=False):
        return self.forward_dotproduct(batch_data)
    
    def graph_dist_loss(self, u_item_dist, u_item_emb, u_item_mask) :
        #u_item_dist #batch, prev_item_count + 1, prev_count + 1
        #u_item_emb #batch, prev_item_count, emb
        #u_item_mask #batch, prev_item_count
        u_item_mask = u_item_mask.float().unsqueeze(2)#batch, prev_item_count, 1
        u_item_dist = u_item_dist[:,1:,1:]#batch, prev_item_count, prev_count
        u_item_mat_mask = torch.bmm(u_item_mask, u_item_mask.permute(0, 2, 1))#batch, prev_item_count, prev_item_count
        u_item_lin_emb = u_item_emb.unsqueeze(1).repeat(1, self.args.uprev_review_limit, 1, 1)#batch, prev_item_count, prev_count, emb
        u_item_col_emb = u_item_emb.unsqueeze(2).repeat(1, 1, self.args.uprev_review_limit, 1)#batch, prev_item_count, prev_count, emb
        u_item_mat_emb = torch.cat([u_item_lin_emb,u_item_col_emb,u_item_lin_emb * u_item_col_emb], dim = 3)#batch, prev_item_count, prev_count, emb * 3
        
        u_item_pred_dist = self.dist_layer(u_item_mat_emb).squeeze(3)#batch, prev_item_count, prev_item_count
        #print(u_item_pred_dist)
        #print(u_item_dist)
        #print(u_item_mat_mask)
        upper_mask = torch.triu(torch.ones(self.args.uprev_review_limit, self.args.uprev_review_limit, device=u_item_mask.device))
        loss = (u_item_pred_dist - u_item_dist) ** 2 * u_item_mat_mask * upper_mask
        loss = torch.sum(loss, [1,2]).mean()

        return loss
    
    def seq_decode_loss(self, user_profile, u_item_emb, init_emb, u_item_mask, add_n_negs = 5) :
        #user_profile #batch_size, emb
        #u_item_emb #batch_size, prev_item_count, emb
        #init_emb #batch_size, 1, emb
        #u_item_mask #batch_size, prev_item_count
        batch_size, _ = user_profile.size()
        h = user_profile
        init_emb = init_emb.squeeze(1)
        h = self.decoder(init_emb,h)#batch, emb
        output = self.decoder_mlp(h).unsqueeze(1)#batch, 1, emb
        for i in range(1, self.args.uprev_review_limit) : 
            h = self.decoder(output[:,i - 1,:],h)
            h_out = self.decoder_mlp(h)
            output = torch.cat([output,h_out.unsqueeze(1)], dim = 1)#batch, i + 1, emb
        #output batch, prev_item_count, emb
        
        neg_sample_idxs = torch.multinomial(self.prod_dists, batch_size * add_n_negs, replacement=True)
        neg_sample_emb = self.product_emb(neg_sample_idxs).view(-1, add_n_negs, self.embedding_size)
        u_item_emb = torch.cat([u_item_emb, neg_sample_emb], dim = 1)#batch_size, prev_item_count + add_n_negs, emb
        
        cos_sim = torch.bmm(output, u_item_emb.permute(0, 2, 1))#batch, prev_item_count, prev_item_count + add_n_negs
        cos_sim = cos_sim.view(-1, self.args.uprev_review_limit + add_n_negs)
        label = torch.tensor(np.arange(1, self.args.uprev_review_limit + 1), dtype = torch.long, device=u_item_mask.device)
        label = label.repeat(batch_size).view(-1, self.args.uprev_review_limit) * u_item_mask - 1 #batch, prev_item_count
        label = label.view(-1)
        loss = self.decoder_loss(cos_sim, label)
        
        return loss
    


    def forward_dotproduct(self, batch_data, train_pv=False):
        query_word_idxs = batch_data.query_word_idxs
        target_prod_idxs = batch_data.target_prod_idxs
        u_item_idxs = batch_data.u_item_idxs
        u_item_query_word_idxs = batch_data.u_item_query_word_idxs#batch, prev_item_count, max_query_len
        u_item_dist = batch_data.u_item_dist#batch, prev_item_count + 1, prev_item_count + 1

        batch_size, prev_item_count = u_item_idxs.size()
        neg_k = self.args.neg_per_pos
        neg_item_idxs = batch_data.neg_prod_idxs
        pos_item_dist = batch_data.target_prod_dist
        neg_item_dist = batch_data.neg_prod_dist
        
        query_word_emb = self.word_embeddings(query_word_idxs)
        query_org_emb = self.query_encoder(query_word_emb, query_word_idxs.ne(self.word_pad_idx))
        query_emb = query_org_emb.unsqueeze(1)

        u_item_query_word_idxs = u_item_query_word_idxs.view(batch_size * prev_item_count, -1)
        u_item_query_word_emb = self.word_embeddings(u_item_query_word_idxs)#batch * prev_item_count, max_query_len, emb
        u_item_query_emb = self.query_encoder(u_item_query_word_emb, u_item_query_word_idxs.ne(self.word_pad_idx)).view(batch_size, prev_item_count, self.embedding_size)#batch, prev_item_count, emb
        
        u_item_query_emb = torch.cat([query_emb, u_item_query_emb], dim=1)#batch, (prev_item_count+1), emb
        u_item_sim_matrix = torch.cosine_similarity(u_item_query_emb.unsqueeze(1), u_item_query_emb.unsqueeze(2), dim=-1).view(-1, prev_item_count+1, prev_item_count+1)#batch, prev_item_count+1, prev_item_count+1
        u_item_sim_matrix_G = u_item_sim_matrix.unsqueeze(1).expand(-1, self.args.heads, -1, -1)
        
        u_item_sim_matrix_F = self.F_matrix(u_item_dist.view(-1, 1)).view(-1, prev_item_count+1, prev_item_count+1, self.args.heads).permute(0, 3, 1, 2)
        u_item_sim_matrix_F[:,:,0,:] = 0.0
        u_item_sim_matrix_F[:,:,:,0] = 0.0
        u_item_sim_matrix = u_item_sim_matrix_G + u_item_sim_matrix_F #
        
        column_mask = torch.ones(batch_size, 1, dtype=torch.uint8, device=query_word_idxs.device)
        u_item_mask = u_item_idxs.ne(self.prod_pad_idx)

        item_seq_mask = torch.cat([column_mask, u_item_mask], dim=1) #batch_size, 1+max_review_count

        target_item_emb = self.product_emb(target_prod_idxs)
        neg_item_emb = self.product_emb(neg_item_idxs) #batch_size, neg_k, embedding_size

        u_item_emb = self.product_emb(u_item_idxs)
        cls_emb = torch.zeros(batch_size, 1, self.embedding_size, dtype=torch.float, device=query_word_idxs.device)
        sequence_emb = torch.cat([cls_emb, u_item_emb], dim=1)


        out_pos = -1 if self.args.use_item_pos else 0
        top_vecs = self.transformer_encoder.encode(sequence_emb, item_seq_mask, u_item_sim_matrix, use_pos=self.args.use_pos_emb)
        
        pos_out_emb = top_vecs[:,out_pos,:] #batch_size, embedding_size
        pos_query_emb = query_org_emb#batch_size, embedding_size
        item_out_emb = top_vecs[:,1:,:]#batch, prev_item_count, embedding_size

        dist_loss = self.graph_dist_loss(u_item_dist, item_out_emb, u_item_mask)
        decode_loss = self.seq_decode_loss(pos_out_emb, u_item_emb, cls_emb, u_item_mask)
        
        pos_u_match_scores = torch.bmm(pos_out_emb.unsqueeze(1), target_item_emb.unsqueeze(2)).view(-1, 1) #in case batch_size=1
        pos_q_match_scores = torch.bmm(pos_query_emb.unsqueeze(1), target_item_emb.unsqueeze(2)).view(-1, 1) #in case batch_size=1
        pos_match_scores = self.match_lambda * pos_u_match_scores + (1 - self.match_lambda) * pos_q_match_scores
        pos_feature_scores = self.feature_layer(pos_item_dist.view(-1, prev_item_count + 1)).view(-1, 1)
        pos_scores = self.score_layer(torch.cat([pos_feature_scores, pos_match_scores], dim=1)).view(batch_size)

        neg_out_emb = top_vecs[:,out_pos,:]#batch_size,embedding_size
        neg_out_emb = neg_out_emb.repeat(1, neg_k).view(-1, self.embedding_size)#batch_size*neg_k, embedding_size
        neg_query_emb = query_org_emb.repeat(1, neg_k).view(-1, self.embedding_size)#batch_size*neg_k, embedding_size

        neg_u_match_scores = torch.bmm(neg_out_emb.unsqueeze(1), neg_item_emb.view(batch_size*neg_k, -1).unsqueeze(2)).view(-1, 1)
        neg_q_match_scores = torch.bmm(neg_query_emb.unsqueeze(1), neg_item_emb.view(batch_size*neg_k, -1).unsqueeze(2)).view(-1, 1)
        neg_match_scores = self.match_lambda * neg_u_match_scores + (1 - self.match_lambda) * neg_q_match_scores
        neg_feature_scores = self.feature_layer(neg_item_dist.view(-1, prev_item_count + 1)).view(-1, 1)
        neg_scores = self.score_layer(torch.cat([neg_feature_scores, neg_match_scores], dim=1)).view(batch_size, neg_k)


        if self.args.sim_func == "bias_product":
            pos_bias = self.product_bias[target_prod_idxs.view(-1)].view(batch_size)
            neg_bias = self.product_bias[neg_item_idxs.view(-1)].view(batch_size, neg_k)
            pos_scores += pos_bias
            neg_scores += neg_bias
        pos_weight = 1
        if self.args.pos_weight:
            pos_weight = self.args.neg_per_pos
        prod_mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.uint8, device=query_word_idxs.device) * pos_weight,
                        torch.ones(batch_size, neg_k, dtype=torch.uint8, device=query_word_idxs.device)], dim=-1)
        prod_scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)
        target = torch.cat([torch.ones(batch_size, 1, device=query_word_idxs.device),
            torch.zeros(batch_size, neg_k, device=query_word_idxs.device)], dim=-1)

        ps_loss = nn.functional.binary_cross_entropy_with_logits(
                prod_scores, target,
                weight=prod_mask.float(),
                reduction='none')
        ps_loss = ps_loss.sum(-1).mean()
        
        self.ps_loss += ps_loss.item()
        self.dist_loss += dist_loss.item()
        self.decode_loss += decode_loss.item()

        return ps_loss + self.alpha * dist_loss + self.beta * decode_loss

    
    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(" ItemTransformerRanker initialization started.")
        if self.pretrain_emb_dir is None:
            nn.init.normal_(self.word_embeddings.weight)
        self.query_encoder.initialize_parameters(logger)
        if self.args.model_name == "item_transformer":
            self.transformer_encoder.initialize_parameters(logger)
        if logger:
            logger.info(" ItemTransformerRanker initialization finished.")

