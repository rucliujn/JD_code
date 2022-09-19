from tqdm import tqdm
from others.logging import logger

import shutil
import torch
import torch.nn.utils as utils
import numpy as np
import data
import os
import time
import sys

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

class Trainer(object):
    """
    Class that controls the training process.
    """
    def __init__(self,  args, model,  optim):
        # Basic attributes.
        self.args = args
        self.model = model
        self.optim = optim
        if (model):
            n_params = _tally_parameters(model)
            logger.info('* number of parameters: %d' % n_params)
        #self.device = "cpu" if self.n_gpu == 0 else "cuda"

        self.ExpDataset = data.ItemPVDataset
        self.ExpDataloader = data.ItemPVDataloader

    def train(self, args, global_data, train_prod_data, valid_prod_data):
        """
        The main training loops.
        """
        logger.info('Start training...')

        model_dir = args.save_dir
        valid_dataset = self.ExpDataset(args, global_data, valid_prod_data)
        step_time, loss = 0.,0.
        get_batch_time = 0.0
        start_time = time.time()
        current_step = 0
        best_mrr = 0.
        step = 0
        best_checkpoint_path = ''
        for current_epoch in range(args.start_epoch+1, args.max_train_epoch+1):
            self.model.train()
            logger.info("Initialize epoch:%d" % current_epoch)
            train_prod_data.initialize_epoch()
            dataset = self.ExpDataset(args, global_data, train_prod_data)
            training_step = len(dataset) / args.batch_size * 10
            prepare_pv = current_epoch < args.train_pv_epoch+1
            print(prepare_pv)
            dataloader = self.ExpDataloader(
                    args, dataset, prepare_pv=prepare_pv, batch_size=args.batch_size,
                    shuffle=True, num_workers=args.num_workers)
            pbar = tqdm(dataloader)
            pbar.set_description("[Epoch {}]".format(current_epoch))
            time_flag = time.time()
            if (current_epoch == args.start_epoch + 11) :
                step = 0

            for batch_data_arr in pbar:
                step += 1
                if batch_data_arr is None:
                    continue
                if type(batch_data_arr) is list:
                    batch_data_arr = [x.to(args.device) for x in batch_data_arr]
                else:
                    batch_data_arr = [batch_data_arr.to(args.device)]
                for batch_data in batch_data_arr:
                    get_batch_time += time.time() - time_flag
                    time_flag = time.time()
                    step_loss = self.model(batch_data, train_pv=prepare_pv)
                    #self.optim.optimizer.zero_grad()
                    self.model.zero_grad()
                    step_loss.backward()
                    utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    
                    #print(current_epoch, args.start_epoch + 1, step, training_step)
                    if (current_epoch <= args.start_epoch + 10) :
                        self.optim.optimizer.param_groups[0]['lr'] = self.args.lr 
                    else :
                        self.optim.optimizer.param_groups[0]['lr'] = (1 - (step * 1.0 / training_step)) * self.args.lr 
                    
                    self.optim.optimizer.step()
                    
                    step_loss = step_loss.item()
                    pbar.set_postfix(step_loss=step_loss, lr=self.optim.optimizer.param_groups[0]['lr'])
                    loss += step_loss / args.steps_per_checkpoint #convert an tensor with dim 0 to value
                    current_step += 1
                    step_time += time.time() - time_flag

                    # Once in a while, we print statistics.
                    if current_step % args.steps_per_checkpoint == 0:
                        ps_loss, item_loss = 0, 0
                        if hasattr(self.model, "ps_loss"):
                            ps_loss = self.model.ps_loss/args.steps_per_checkpoint
                        if hasattr(self.model, "dist_loss"):
                            dist_loss = self.model.dist_loss/args.steps_per_checkpoint
                        if hasattr(self.model, "decode_loss"):
                            decode_loss = self.model.decode_loss/args.steps_per_checkpoint

                        logger.info("Epoch %d lr = %5.6f loss = %6.2f ps_loss: %3.2f dist_loss: %3.2f decode_loss: %3.2f time %.2f prepare_time %.2f step_time %.2f" %
                                (current_epoch, self.optim.optimizer.param_groups[0]['lr'], loss, ps_loss, dist_loss, decode_loss, 
                                    time.time()-start_time, get_batch_time, step_time))#, end=""
                        step_time, get_batch_time, loss = 0., 0.,0.
                        if hasattr(self.model, "ps_loss"):
                            self.model.clear_loss()
                        sys.stdout.flush()
                        start_time = time.time()
            checkpoint_path = os.path.join(model_dir, 'model_epoch_%d.ckpt' % current_epoch)
            self._save(current_epoch, checkpoint_path)
            mrr, prec = self.validate(args, global_data, valid_dataset)
            logger.info("Epoch {}: MRR:{} P@1:{}".format(current_epoch, mrr, prec))
            if mrr > best_mrr:
                best_mrr = mrr
                best_checkpoint_path = os.path.join(model_dir, 'model_best.ckpt')
                logger.info("Copying %s to checkpoint %s" % (checkpoint_path, best_checkpoint_path))
                shutil.copyfile(checkpoint_path, best_checkpoint_path)
        return best_checkpoint_path

    def _save(self, epoch, checkpoint_path):
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'opt': self.args,
            'optim': self.optim,
        }
        #model_dir = "%s/model" % (self.args.save_dir)
        #checkpoint_path = os.path.join(model_dir, 'model_epoch_%d.ckpt' % epoch)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        torch.save(checkpoint, checkpoint_path)

    def validate(self, args, global_data, valid_dataset):
        """ Validate model.
        """
        prod_pad = global_data.product_size
        dataloader = self.ExpDataloader(
                args, valid_dataset, batch_size=args.valid_batch_size,
                shuffle=False, num_workers=args.num_workers)
        all_prod_idxs, all_prod_scores, all_target_idxs, \
                all_query_idxs, all_user_idxs \
                = self.get_prod_scores(args, global_data, valid_dataset, dataloader, "Validation")
        mrr, prec = self.calc_metrics(all_prod_idxs, all_prod_scores, all_user_idxs, all_target_idxs, prod_pad, cutoff=100)
        return mrr, prec

    def test(self, args, global_data, test_prod_data, rankfname="test.best_model.ranklist", cutoff=100):
        test_dataset = self.ExpDataset(args, global_data, test_prod_data)
        dataloader = self.ExpDataloader(
                args, test_dataset, batch_size=args.valid_batch_size, #batch_size
                shuffle=False, num_workers=args.num_workers)

        all_prod_idxs, all_prod_scores, all_target_idxs, \
                all_query_idxs, all_user_idxs \
                = self.get_prod_scores(args, global_data, test_dataset, dataloader, "Test")
        
        mrr, prec = self.calc_metrics(all_prod_idxs, all_prod_scores, all_user_idxs, all_target_idxs, prod_pad, cutoff=100)
        whole_length = all_prod_idxs.shape[0]
        logger.info("Test: MRR:{} P@1:{}".format(mrr, prec))
        output_path = os.path.join(args.save_dir, rankfname)

        with open(output_path, 'w') as rank_fout:
            last_user_idx = -1
            prod_idx_scores = []

            for i in range(whole_length) :
                if (all_user_idxs[i] != last_user_idx) :
                    if (prod_idx_scores != []) :
                        prod_idx_scores.sort(key = lambda x : -x[1])
                        length = len(prod_idx_scores)
                        for rank in range(length):
                            product_id = global_data.product_ids[prod_idx_scores[rank][0]]
                            score = prod_idx_scores[rank][1]
                            line = "%s_%d Q0 %s %d %f ReviewTransformer\n" \
                                % (all_user_idxs[i], 1, product_id, rank+1, score)
                            rank_fout.write(line)
                    
                    last_user_idx = all_user_idxs[i]
                    prod_idx_scores = []
                
                if (all_prod_idxs[i] != global_data.product_size) :
                    prod_idx_scores.append((all_prod_idxs[i], all_prod_scores))
            
            prod_idx_scores.sort(key = lambda x : -x[1])
            length = len(prod_idx_scores)
            for rank in range(length):
                product_id = global_data.product_ids[prod_idx_scores[rank][0]]
                score = prod_idx_scores[rank][1]
                line = "%s_%d Q0 %s %d %f ReviewTransformer\n" \
                    % (all_user_idxs[i], 1, product_id, rank+1, score)
                rank_fout.write(line)


    def calc_metrics(self, all_prod_idxs, all_prod_scores, all_user_idxs, all_target_idxs, prod_pad, cutoff=100):
        whole_length = all_prod_idxs.shape[0]
        last_user_idx = -1
        prod_idx_scores = []
        eval_count = 0
        mrr, prec = 0, 0
        for i in range(whole_length) :
            if (all_user_idxs[i] != last_user_idx) :
                if (prod_idx_scores != []) :
                    prod_idx_scores.sort(key = lambda x : -x[1])
                    target = all_target_idxs[i - 1]

                    length = len(prod_idx_scores)
                    rank = 0
                    for k in range(length) :
                        if (prod_idx_scores[k][0] == target) :
                            rank = k + 1
                            break
                    if rank != 0 :
                        mrr += 1 / (rank * 1.0)
                    if rank == 1:
                        prec +=1
                
                    eval_count += 1
                    last_user_idx = all_user_idxs[i]
                    prod_idx_scores = []

            if (all_prod_idxs[i] != prod_pad) :
                prod_idx_scores.append((all_prod_idxs[i], all_prod_scores))
        

        prod_idx_scores.sort(key = lambda x : -x[1])
        target = all_target_idxs[-1]

        length = len(prod_idx_scores)
        rank = 0
        for k in range(length) :
            if (prod_idx_scores[k][0] == target) :
                rank = k + 1
                break
        if rank != 0 :
            mrr += 1 / (rank * 1.0)
        if rank == 1:
            prec +=1

    
        mrr /= eval_count
        prec /= eval_count
        print("MRR:{} P@1:{}".format(mrr, prec))
        return mrr, prec

    def get_prod_scores(self, args, global_data, dataset, dataloader, description):
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(dataloader)
            pbar.set_description(description)
            all_prod_scores, all_target_idxs, all_prod_idxs = [], [], []
            all_user_idxs, all_query_idxs = [], []
            for batch_data in pbar:
                batch_data = batch_data.to(args.device)
                batch_scores = self.model.test(batch_data)

                all_user_idxs.append(np.asarray(batch_data.user_idxs))
                all_query_idxs.append(np.asarray(batch_data.query_idxs))
                candi_prod_idxs = batch_data.candi_prod_idxs
                if type(candi_prod_idxs) is torch.Tensor:
                    candi_prod_idxs = candi_prod_idxs.cpu()
                all_prod_idxs.append(np.asarray(candi_prod_idxs))
                all_prod_scores.append(batch_scores.cpu().numpy())
                target_prod_idxs = batch_data.target_prod_idxs
                if type(target_prod_idxs) is torch.Tensor:
                    target_prod_idxs = target_prod_idxs.cpu()
                all_target_idxs.append(np.asarray(target_prod_idxs))
                #use MRR
        all_prod_idxs = np.concatenate(all_prod_idxs, axis=0)
        all_prod_scores = np.concatenate(all_prod_scores, axis=0)
        all_target_idxs = np.concatenate(all_target_idxs, axis=0)
        all_user_idxs = np.concatenate(all_user_idxs, axis=0)
        all_query_idxs = np.concatenate(all_query_idxs, axis=0)
        return all_prod_idxs, all_prod_scores, all_target_idxs, all_query_idxs, all_user_idxs

