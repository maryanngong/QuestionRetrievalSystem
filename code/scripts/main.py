import argparse
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import models.model_utils as model_utils
import train.train_utils as train_utils
import os
import torch
import datetime
import cPickle as pickle
import pdb
import data.myio as myio



parser = argparse.ArgumentParser(description='PyTorch Example Sentiment Classifier')
# learning
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training [default: 64]')
# data loading
parser.add_argument('--num_workers', nargs='?', type=int, default=4, help='num workers for data loader')
# model
parser.add_argument('--model_name', nargs="?", type=str, default='dan', help="Form of model, i.e dan, rnn, etc.")
parser.add_argument('--num_hidden', type=int, default=32, help="encoding size.")
parser.add_argument('--dropout', type=float, default=0.0, help="dropout parameter")
parser.add_argument('--margin', type=float, default=1.0)
#cnn
parser.add_argument('--num_channels', type=int, default=5, help="Number of channels for CNN model aka depth?")
parser.add_argument('--filter_width', type=int, default=3, help="width dimension of CNN filter")
parser.add_argument('--use_mean_pooling', action='store_true', default=False, help="type of pooling to use for CNN")
# device
parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')
parser.add_argument('--cuda_device', type=int, default=0, help='specify GPU number to use')
parser.add_argument('--train', action='store_true', default=False, help='enable train')
# task
parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
parser.add_argument('--save_path', type=str, default="", help='Path where to dump model')

parser.add_argument('--corpus', type=str, default='../../askubuntu/text_tokenized.txt.gz')
parser.add_argument('--android', action='store_true', default=False, help="run evaluation on android dataset")

args = parser.parse_args()


if __name__ == '__main__':
    # update args and print

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    if args.android:
        embeddings, word_to_indx = myio.getGloveEmbeddingTensor()
    else:
        embeddings, word_to_indx = myio.getEmbeddingTensor()
    raw_corpus = myio.read_corpus(args.corpus)
    ids_corpus = myio.map_corpus(raw_corpus, word_to_indx, max_len=100)

    # model
    if args.snapshot is None:
        model = model_utils.get_model(embeddings, args)
    else :
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            model = torch.load(args.snapshot)
        except :
            print("Sorry, This snapshot doesn't exist."); exit()
    print(model)

    if args.android:
        raw_android_corpus = myio.read_corpus('../../Android/corpus.tsv.gz')
        ids_android_corpus = myio.map_corpus(raw_android_corpus, word_to_indx, max_len=100)
        dev_neg_dict = myio.read_annotations_android('../../Android/dev.neg.txt')
        dev_pos_dict = myio.read_annotations_android('../../Android/dev.pos.txt')
        test_neg_dict = myio.read_annotations_android('../../Android/test.neg.txt')
        test_pos_dict = myio.read_annotations_android('../../Android/test.pos.txt')
        dev = myio.create_eval_batches_android(ids_android_corpus, dev_pos_dict, dev_neg_dict)
        test = myio.create_eval_batches_android(ids_android_corpus, test_pos_dict, test_neg_dict)
    else:
        dev = myio.read_annotations('../../askubuntu/dev.txt', K_neg=-1, prune_pos_cnt=-1)
        dev = myio.create_eval_batches(ids_corpus, dev, 0, pad_left=False)
        test = myio.read_annotations('../../askubuntu/test.txt', K_neg=-1, prune_pos_cnt=-1)
        test = myio.create_eval_batches(ids_corpus, test, 0, pad_left=False)

    if args.train :
        train = myio.read_annotations('../../askubuntu/train_random.txt')
        train_utils.train_model(model, train, dev, test, ids_corpus, args.batch_size, args)

    else:
        print("Evaluating performance on dev data...")
        MAP, MRR, P1, P5, auc5 = train_utils.evaluate(dev, model, args)
        print(tabulate([[MAP, MRR, P1, P5, auc5]], headers=['MAP', 'MRR', 'P@1', 'P@5', 'AUC0.05']))

        print()
        print("Evaluating performance on test data...")
        MAP, MRR, P1, P5, auc5 = train_utils.evaluate(test, model, args)
        print(tabulate([[MAP, MRR, P1, P5, auc5]], headers=['MAP', 'MRR', 'P@1', 'P@5', 'AUC0.05']))
        
