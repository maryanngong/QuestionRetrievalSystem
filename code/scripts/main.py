import argparse
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import data.dataset_utils as data_utils
import models.model_utils as model_utils
import train.train_utils as train_utils
import os
import torch
import datetime
import cPickle as pickle
import pdb



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
parser.add_argument('--save_path', type=str, default="model.pt", help='Path where to dump model')
# debugging
parser.add_argument('--debug_mode', action='store_true', default=False, help='debugging mode so work with way less data')

args = parser.parse_args()


if __name__ == '__main__':
    # update args and print

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    use_cnn = False
    if 'cnn' in args.model_name:
        use_cnn=True
    data = data_utils.Dataset(batch_size=args.batch_size, cnn=use_cnn)

    # model
    if args.snapshot is None:
        model = model_utils.get_model(data.get_embeddings(), args)
    else :
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            model = torch.load(args.snapshot)
        except :
            print("Sorry, This snapshot doesn't exist."); exit()
    print(model)

    print()
    # train
    if args.train :
        train_utils.train_model(data, data.create_eval_batches('dev'), model, args)
    else:
        print("Evaluating performance on train data...")
        train_utils.eval_model_two(data.create_eval_batches_train(), model, args)
        # print()
        print("Evaluating performance on dev data...")
        train_utils.eval_model_two(data.create_eval_batches("dev"), model, args)
        print()
        print("Evaluating performance on test data...")
        train_utils.eval_model_two(data.create_eval_batches("test"), model, args)
