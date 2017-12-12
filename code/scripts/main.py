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
from sklearn.feature_extraction.text import TfidfVectorizer
from tabulate import tabulate
import random
from multiprocessing import Process, Lock
import copy
import multiprocessing


def main(args, results_lock=None):
    # update args and print

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    askubuntu_corpus = '../../askubuntu/text_tokenized.txt.gz'

    if args.model_name == 'tfidf':
        if args.android:
            raw_corpus = myio.read_corpus_documents('../../Android/corpus.tsv.gz')
            flat_corpus_text = myio.read_corpus_flat('../../Android/corpus.tsv.gz')
            vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,1), binary=False)
            vectorizer.fit(flat_corpus_text)
            raw_android_corpus = myio.read_corpus_documents('../../Android/corpus.tsv.gz')
            dev_neg_dict = myio.read_annotations_android('../../Android/dev.neg.txt')
            dev_pos_dict = myio.read_annotations_android('../../Android/dev.pos.txt')
            test_neg_dict = myio.read_annotations_android('../../Android/test.neg.txt')
            test_pos_dict = myio.read_annotations_android('../../Android/test.pos.txt')
            dev = myio.create_tfidf_batches_android(raw_android_corpus, dev_pos_dict, dev_neg_dict)
            test = myio.create_tfidf_batches_android(raw_android_corpus, test_pos_dict, test_neg_dict)
        else:
            raw_corpus = myio.read_corpus_documents(askubuntu_corpus)
            flat_corpus_text = myio.read_corpus_flat(askubuntu_corpus)
            vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,1), binary=False)
            vectorizer.fit(flat_corpus_text)
            dev = myio.read_annotations('../../askubuntu/dev.txt', K_neg=-1, prune_pos_cnt=-1)
            dev = myio.create_tfidf_batches(raw_corpus, dev)
            test = myio.read_annotations('../../askubuntu/test.txt', K_neg=-1, prune_pos_cnt=-1)
            test = myio.create_tfidf_batches(raw_corpus, test)
        print("Evaluating performance on dev data...")
        MAP, MRR, P1, P5, auc5 = train_utils.evaluate(vectorizer=vectorizer, vectorizer_data=dev)
        print(tabulate([[MAP, MRR, P1, P5, auc5]], headers=['MAP', 'MRR', 'P@1', 'P@5', 'AUC0.05']))
        print()
        print("Evaluating performance on test data...")
        MAP, MRR, P1, P5, auc5 = train_utils.evaluate(vectorizer=vectorizer, vectorizer_data=test)
        print(tabulate([[MAP, MRR, P1, P5, auc5]], headers=['MAP', 'MRR', 'P@1', 'P@5', 'AUC0.05']))
        print()

    else:
        if args.android:
            embeddings, word_to_indx = myio.getGloveEmbeddingTensor(prune=True, cased=args.cased)
        else:
            embeddings, word_to_indx = myio.getEmbeddingTensor(args.embeddings_path)
        raw_corpus = myio.read_corpus(askubuntu_corpus)
        ids_corpus = myio.map_corpus(raw_corpus, word_to_indx, max_len=100)

        # model
        if args.snapshot is None:
            model = model_utils.get_model(embeddings, args)
            if args.domain_adaptation:
                model_2 = model_utils.get_model(None, args, True)
        else :
            print('\nLoading model from [%s]...' % args.snapshot)
            try:
                model = torch.load(args.snapshot)
                if args.domain_adaptation:
                    model_2 = torch.load(args.snapshot_2)
            except :
                print("Sorry, This snapshot doesn't exist."); exit()
        print(model)
        if args.domain_adaptation:
            print(model_2)

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
            # Create Batch2 batches
            if args.domain_adaptation:
                train_2 = myio.create_discriminator_batches(ids_corpus, ids_android_corpus, (len(train) / args.batch_size + 1))
                train_3 = myio.create_random_negative_batches(ids_android_corpus, (len(train) / args.batch_size + 1))
                train_utils.train_model(model, train, dev, test, ids_corpus, args.batch_size, args, model_2, train_2, results_lock, train_3)
            else:
                train_utils.train_model(model, train, dev, test, ids_corpus, args.batch_size, args)

        else:
            print("Evaluating performance on dev data...")
            MAP, MRR, P1, P5, auc5 = train_utils.evaluate(model_data=dev, model=model, args=args)
            print(tabulate([[MAP, MRR, P1, P5, auc5]], headers=['MAP', 'MRR', 'P@1', 'P@5', 'AUC0.05']))

            print()
            print("Evaluating performance on test data...")
            MAP, MRR, P1, P5, auc5 = train_utils.evaluate(model_data=test, model=model, args=args)
            print(tabulate([[MAP, MRR, P1, P5, auc5]], headers=['MAP', 'MRR', 'P@1', 'P@5', 'AUC0.05']))

def try_random_params(args, gpu, results_lock):
    args_dict = vars(process_args)
    args_dict['gpuid'] = gpu
    for i in range(1000):
        for p in tunable_params:
            if isinstance(param_specs[p][1], int):
                args_dict[p] = random.randint(param_specs[p][0], param_specs[p][1])
            else:
                args_dict[p] = random.uniform(param_specs[p][0], param_specs[p][1])
        main(args, results_lock)
    print("DONE!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Question Retrieval Model')
    # task
    parser.add_argument('-d', '--domain_adaptation', action='store_true', default=False, help='choose adaptation transfer setting')
    # learning
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--lr2', type=float, default=-0.001, help='initial learning rate [default: -0.001]')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs for train [default: 256]')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training [default: 64]')
    parser.add_argument('--lam', type=float, default=0.0001, help='constant multiplier on loss 2 in domain adaptation')
    parser.add_argument('--gam', type=float, default=0.0001, help='constant multiplier on loss 3 in domain adaptation')
    parser.add_argument('--noise_factor', type=float, default=0.1)
    # data
    parser.add_argument('--embeddings_path', type=str, default='../../askubuntu/vector/vectors_pruned.200.txt.gz', help='path for word embeddings')
    parser.add_argument('--cased', action='store_true', default=False, help="use cased glove embeddings")
    # model
    parser.add_argument('--model_name', nargs="?", type=str, default='cnn3', choices=['dan', 'cnn2', 'cnn3', 'cnn4', 'lstm_bi', 'lstm_bi_fc', 'lstm3', 'tfidf'], help="Encoder model type [dan, cnn2, cnn3, cnn4, lstm_bi, lstm_bi_fc, lstm3]")
    parser.add_argument('--model_name_2', nargs="?", type=str, default='ffn', choices=['ffn'], help="Discriminator model type")
    parser.add_argument('--num_hidden', type=int, default=32, help="encoding size.")
    parser.add_argument('--dropout', type=float, default=0.0, help="dropout parameter")
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--margin_b', type=float, default=1.0)
    # device
    parser.add_argument('-c', '--cuda', action='store_true', default=False, help='enable the gpu')
    parser.add_argument('-t', '--train', action='store_true', default=False, help='enable train')
    # task
    parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
    parser.add_argument('--snapshot2', type=str, default=None, help='filename of discriminator model snapshot to load[default: None]')
    parser.add_argument('--save_path', type=str, default="", help='Path where to dump model')
    parser.add_argument('--results_path', type=str, default="all_results.txt", help="Path where to save best results")

    parser.add_argument('-a', '--android', action='store_true', default=False, help="run evaluation on android dataset")
    parser.add_argument('--gpuid', type=int, default=0, help="set cuda device for torch")
    parser.add_argument('--hyperparam_search', action='store_true', default=False, help="search over possible hp combos")

    parser.add_argument('--show_discr_loss', action='store_true', default=False, help="print out discriminator loss and accuracy each batch")

    args = parser.parse_args()


    if args.hyperparam_search:
        processes = []
        manager = multiprocessing.Manager()
        results_lock = Lock()
        tunable_params = ['lr', 'lam', 'num_hidden', 'dropout', 'margin']
        # (minval, maxval)
        param_specs = {'lr':(0.0001, 0.01), 'lam':(0.0000001, 0.001), 'num_hidden':(300, 700), 'dropout':(0.0, 0.4), 'margin':(0.2, 0.8)}
        random_params = []
        for gpu in range(4):
            process_args = manager.Namespace(**(copy.deepcopy(vars(args))))
            processes.append(Process(target=try_random_params, args=(process_args, gpu, results_lock)))
            processes[len(processes)-1].start()
        for process in processes:
            process.join()
    else:
        main(args)
