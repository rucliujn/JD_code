import torch



class ItemPVBatch(object):
    def __init__(self, query_word_idxs, target_prod_idxs, u_item_idxs, u_item_query_word_idxs, u_item_dist, query_idxs = [], user_idxs =[], 
                candi_prod_idxs=[], neg_prod_idxs = [], target_prod_dist = [], neg_prod_dist = [], candi_prod_dist = [], to_tensor=True): #"cpu" or "cuda"
        self.query_word_idxs = query_word_idxs
        self.target_prod_idxs = target_prod_idxs
        self.u_item_idxs = u_item_idxs
        self.u_item_query_word_idxs = u_item_query_word_idxs
        self.u_item_dist = u_item_dist

        self.query_idxs = query_idxs
        self.user_idxs = user_idxs

        self.candi_prod_idxs = candi_prod_idxs
        self.neg_prod_idxs = neg_prod_idxs

        self.target_prod_dist = target_prod_dist
        self.neg_prod_dist = neg_prod_dist
        self.candi_prod_dist = candi_prod_dist

        if to_tensor:
            self.to_tensor()

    def to_tensor(self):
        self.query_word_idxs = torch.tensor(self.query_word_idxs)
        self.target_prod_idxs = torch.tensor(self.target_prod_idxs)
        self.u_item_idxs = torch.tensor(self.u_item_idxs)
        self.u_item_query_word_idxs = torch.tensor(self.u_item_query_word_idxs)
        self.u_item_dist = torch.tensor(self.u_item_dist, dtype=torch.float)
        
        self.candi_prod_idxs = torch.tensor(self.candi_prod_idxs)
        self.neg_prod_idxs = torch.tensor(self.neg_prod_idxs)

        self.target_prod_dist = torch.tensor(self.target_prod_dist, dtype=torch.float)
        self.neg_prod_dist = torch.tensor(self.neg_prod_dist, dtype=torch.float)
        self.candi_prod_dist = torch.tensor(self.candi_prod_dist, dtype=torch.float)

    def to(self, device):
        if device == "cpu":
            return self
        else:
            query_word_idxs = self.query_word_idxs.to(device)
            target_prod_idxs = self.target_prod_idxs.to(device)
            u_item_idxs = self.u_item_idxs.to(device)
            u_item_query_word_idxs = self.u_item_query_word_idxs.to(device)
            u_item_dist = self.u_item_dist.to(device)

            candi_prod_idxs = self.candi_prod_idxs.to(device)
            neg_prod_idxs = self.neg_prod_idxs.to(device)
            
            target_prod_dist = self.target_prod_dist.to(device)
            neg_prod_dist = self.neg_prod_dist.to(device)
            candi_prod_dist = self.candi_prod_dist.to(device)

            return self.__class__(
                    query_word_idxs, target_prod_idxs, u_item_idxs, u_item_query_word_idxs, u_item_dist, self.query_idxs, self.user_idxs,
                    candi_prod_idxs, neg_prod_idxs, target_prod_dist, neg_prod_dist, candi_prod_dist, to_tensor=False)
