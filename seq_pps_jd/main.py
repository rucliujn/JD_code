"""
main entry of the script, train, validate and test
"""
import torch
import argparse
import random
import glob
import os
import numpy as np

from others.logging import logger, init_logger
from models.item_transformer import ItemTransformerRanker
from data.data_util import GlobalProdSearchData, ProdSearchData
from trainer import Trainer
from models.optimizers import Optimizer

seed = 666
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def build_optim(args, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if args.train_from != '' and checkpoint is not None:
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps,
            weight_decay=args.l2_lambda)
        #self.start_decay_steps take effect when decay_method is not noam

    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '' and checkpoint is not None:
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.device == "cuda":
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=666, type=int)
    parser.add_argument("--train_from", default='')
    parser.add_argument("--model_name", default='review_transformer',
            choices=['review_transformer', 'item_transformer', 'ZAM', 'AEM', 'QEM'], help="which type of model is used to train")
    parser.add_argument("--sep_prod_emb", type=str2bool, nargs='?',const=True,default=False,
            help="whether to use separate embeddings for historical product and the target product")
    parser.add_argument("--pretrain_emb_dir", default='', help="pretrained paragraph and word embeddings")
    parser.add_argument("--pretrain_up_emb_dir", default='', help="pretrained user item embeddings")
    #parser.add_argument("--review_train_mode", type=str, default='random', choices=["random", "prev", "last"]
    #        help="use (sorted) random reviews in training data; reviews before current purchase; reviews that occur at the last in training set for training; ")
    parser.add_argument("--fix_train_review", type=str2bool, nargs='?',const=True,default=False,
            help="fix train reviews (the last reviews in the training set); ")
    parser.add_argument("--do_seq_review_train", type=str2bool, nargs='?',const=True,default=False,
            help="only use reviews before current purchase for training; ")
    parser.add_argument("--do_seq_review_test", type=str2bool, nargs='?',const=True,default=False,
            help="during test time if only training data is available, use the most recent iprev and uprev reviews; if train_review_only is False, use all the sequential reviews available before current review, including validation and test.")
    parser.add_argument("--fix_emb", type=str2bool, nargs='?',const=True,default=False,
            help="fix word embeddings or review embeddings during training.")
    parser.add_argument("--use_dot_prod", type=str2bool, nargs='?',const=True,default=True,
            help="use positional embeddings when encoding reviews.")
    parser.add_argument("--sim_func", type=str, default="product", choices=["bias_product", "product", "cosine"], help="similarity computation method.")
    parser.add_argument("--use_pos_emb", type=str2bool, nargs='?',const=True,default=True,
            help="use positional embeddings when encoding reviews.")
    parser.add_argument("--use_seg_emb", type=str2bool, nargs='?',const=True,default=True,
            help="use segment embeddings when encoding reviews.")
    parser.add_argument("--use_item_pos", type=str2bool, nargs='?',const=True,default=False,
            help="use the embeddings corresponding to a candidate item as an output when encoding the purchased item sequence.")
    parser.add_argument("--use_item_emb", type=str2bool, nargs='?',const=True,default=False,
            help="use item embeddings when encoding review sequence.")
    parser.add_argument("--use_user_emb", type=str2bool, nargs='?',const=True,default=False,
            help="use user embeddings when encoding review sequence.")
    parser.add_argument("--rankfname", default="test.best_model.ranklist")
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--token_dropout", default=0.1, type=float)
    parser.add_argument("--optim", type=str, default="adam", help="sgd or adam")
    parser.add_argument("--lr", default=0.002, type=float) #0.002
    parser.add_argument("--beta1", default= 0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--decay_method", default='adam', choices=['noam', 'adam'],type=str) #warmup learning rate then decay
    parser.add_argument("--warmup_steps", default=8000, type=int) #10000
    parser.add_argument("--max_grad_norm", type=float, default=5.0,
                            help="Clip gradients to this norm.")
    parser.add_argument("--subsampling_rate", type=float, default=1e-5,
                            help="The rate to subsampling.")
    parser.add_argument("--do_subsample_mask", type=str2bool, nargs='?',const=True,default=False,
            help="do subsampling mask do the reviews with cutoff review_word_limit; otherwise do subsampling then do the cutoff.")
    parser.add_argument("--prod_freq_neg_sample", type=str2bool, nargs='?',const=True,default=False,
            help="whether to sample negative products according to their purchase frequency.")
    parser.add_argument("--pos_weight", type=str2bool, nargs='?',const=True,default=False,
            help="use pos_weight different from 1 during training.")
    parser.add_argument("--l2_lambda", type=float, default=0.0,
                            help="The lambda for L2 regularization.")
    parser.add_argument("--batch_size", type=int, default=32,
                            help="Batch size to use during training.")
    parser.add_argument("--has_valid", type=str2bool, nargs='?',const=True, default=True,
            help="whether there is validation set; if not use test as validation.")
    parser.add_argument("--valid_batch_size", type=int, default=24,
                            help="Batch size for validation to use during training.")
    parser.add_argument("--valid_candi_size", type=int, default=-1, # can be -1 if you want to evaluate on all the products.
                            help="Random products used for validation. When it is 0 or less, all the products are used.")
    parser.add_argument("--test_candi_size", type=int, default=-1, #
                            help="When it is 0 or less, all the products are used. Otherwise, test_candi_size samples from ranklist will be reranked")
    parser.add_argument("--candi_batch_size", type=int, default=500,
                            help="Batch size for validation to use during training.")
    parser.add_argument("--num_workers", type=int, default=4,
                            help="Number of processes to load batches of data during training.")
    parser.add_argument("--data_dir", type=str, default="/tmp", help="Data directory")
    parser.add_argument("--input_train_dir", type=str, default="", help="The directory of training and testing data")
    parser.add_argument("--save_dir", type=str, default="/tmp", help="Model directory & output directory")
    parser.add_argument("--log_file", type=str, default="train.log", help="log file name")
    parser.add_argument("--query_encoder_name", type=str, default="fs", choices=["fs","avg"],
            help="Specify network structure parameters. Please read readme.txt for details.")
    parser.add_argument("--review_encoder_name", type=str, default="pvc", choices=["pv", "pvc", "fs", "avg"],
            help="Specify network structure parameters. ")
    parser.add_argument("--embedding_size", type=int, default=128, help="Size of each embedding.")
    parser.add_argument("--ff_size", type=int, default=512, help="size of feedforward layers in transformers.")
    parser.add_argument("--heads", default=8, type=int, help="attention heads in transformers")
    parser.add_argument("--inter_layers", default=2, type=int, help="transformer layers")
    parser.add_argument("--review_word_limit", type=int, default=100,
                            help="the limit of number of words in reviews, for review_transformer.")
    parser.add_argument("--uprev_review_limit", type=int, default=20,
                            help="the number of items the user previously purchased in TEM; \
                                    the number of users previous reviews used in RTM.")
    parser.add_argument("--iprev_review_limit", type=int, default=30,
                            help="the number of item's previous reviews used.")
    parser.add_argument("--pv_window_size", type=int, default=1, help="Size of context window.")
    parser.add_argument("--upv_window_size", type=int, default=5, help="Size of context window for history item.")
    parser.add_argument("--corrupt_rate", type=float, default=0.9, help="the corruption rate that is used to represent the paragraph in the corruption module.")
    parser.add_argument("--shuffle_review_words", type=str2bool, nargs='?',const=True,default=True,help="shuffle review words before collecting sliding words.")
    parser.add_argument("--train_review_only", type=str2bool, nargs='?',const=True,default=True,help="whether the representation of negative products need to be learned at each step.")
    parser.add_argument("--max_train_epoch", type=int, default=20,
                            help="Limit on the epochs of training (0: no limit).")
    parser.add_argument("--train_pv_epoch", type=int, default=0,
                            help="Limit on the epochs of training pv (0: do not train according to pv loss).")
    parser.add_argument("--start_epoch", type=int, default=0,
                            help="the epoch where we start training.")
    parser.add_argument("--steps_per_checkpoint", type=int, default=200,
                            help="How many training steps to do per checkpoint.")
    parser.add_argument("--neg_per_pos", type=int, default=5,
                            help="How many negative samples used to pair with postive results.")
    parser.add_argument("--sparse_emb", action='store_true',
                            help="use sparse embedding or not.")
    parser.add_argument("--scale_grad", action='store_true',
                            help="scale the grad of word and av embeddings.")
    parser.add_argument("-nw", "--weight_distort", action='store_true',
                            help="Set to True to use 0.75 power to redistribute for neg sampling .")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--rank_cutoff", type=int, default=100,
                            help="Rank cutoff for output ranklists.")
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help="use CUDA or cpu")
    parser.add_argument("--alpha", type=float, default=0.05,
                            help="The weight for graph decode.")
    parser.add_argument("--beta", type=float, default=0.5,
                            help="The weight for seq decode.")                   
    parser.add_argument("--dist_pad", type=int, default=10,
                            help="The pad for graph decode.") 
    return parser.parse_args()

model_flags = ['embedding_size', 'ff_size', 'heads', 'inter_layers','review_encoder_name','query_encoder_name']

def create_model(args, global_data, prod_data, load_path=''):
    """Create translation model and initialize or load parameters in session."""
    model = ItemTransformerRanker(args, args.device, global_data.vocab_size,
        global_data.product_size,  global_data.words, word_dists=prod_data.word_dists)

    if os.path.exists(load_path):
    #if load_path != '':
        logger.info('Loading checkpoint from %s' % load_path)
        checkpoint = torch.load(load_path,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
        args.start_epoch = checkpoint['epoch']
        model.load_cp(checkpoint)
        optim = build_optim(args, model, checkpoint)
    else:
        logger.info('No available model to load. Build new model.')
        optim = build_optim(args, model, None)
    logger.info(model)
    return model, optim

def train(args):
    args.start_epoch = 0
    logger.info('Device %s' % args.device)

    seed = args.seed    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    global_data = GlobalProdSearchData(args, args.data_dir, args.input_train_dir)
    train_prod_data = ProdSearchData(args, args.input_train_dir, "train", global_data)

    model, optim = create_model(args, global_data, train_prod_data, args.train_from)
    trainer = Trainer(args, model, optim)
    valid_prod_data = ProdSearchData(args, args.input_train_dir, "test", global_data)
    best_checkpoint_path = trainer.train(trainer.args, global_data, train_prod_data, valid_prod_data)
    
    test_prod_data = ProdSearchData(args, args.input_train_dir, "test", global_data)
    best_model, _ = create_model(args, global_data, train_prod_data, best_checkpoint_path)
    del trainer
    torch.cuda.empty_cache()
    trainer = Trainer(args, best_model, None)
    trainer.test(args, global_data, test_prod_data, args.rankfname)


def get_product_scores(args):

    seed = args.seed    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    global_data = GlobalProdSearchData(args, args.data_dir, args.input_train_dir)
    test_prod_data = ProdSearchData(args, args.input_train_dir, "test", global_data)
    model_path = os.path.join(args.save_dir, 'model_best.ckpt')
    best_model, _ = create_model(args, global_data, test_prod_data, model_path)
    trainer = Trainer(args, best_model, None)
    trainer.valid(args, global_data, test_prod_data, args.rankfname)

def main(args):
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    init_logger(os.path.join(args.save_dir, args.log_file))
    logger.info(args)
    if args.mode == "train":
        train(args)
    else:
        get_product_scores(args)
if __name__ == '__main__':
    main(parse_args())
