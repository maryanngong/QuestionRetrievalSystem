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
import data.data_utils as data_utils
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
            raw_corpus = data_utils.read_corpus_documents('../../Android/corpus.tsv.gz')
            flat_corpus_text = data_utils.read_corpus_flat('../../Android/corpus.tsv.gz')
            vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,1), binary=False)
            vectorizer.fit(flat_corpus_text)
            raw_android_corpus = data_utils.read_corpus_documents('../../Android/corpus.tsv.gz')
            dev_neg_dict = data_utils.read_annotations_android('../../Android/dev.neg.txt')
            dev_pos_dict = data_utils.read_annotations_android('../../Android/dev.pos.txt')
            test_neg_dict = data_utils.read_annotations_android('../../Android/test.neg.txt')
            test_pos_dict = data_utils.read_annotations_android('../../Android/test.pos.txt')
            dev = data_utils.create_tfidf_batches_android(raw_android_corpus, dev_pos_dict, dev_neg_dict)
            test = data_utils.create_tfidf_batches_android(raw_android_corpus, test_pos_dict, test_neg_dict)
        else:
            raw_corpus = data_utils.read_corpus_documents(askubuntu_corpus)
            flat_corpus_text = data_utils.read_corpus_flat(askubuntu_corpus)
            vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,1), binary=False)
            vectorizer.fit(flat_corpus_text)
            dev = data_utils.read_annotations('../../askubuntu/dev.txt', K_neg=-1, prune_pos_cnt=-1)
            dev = data_utils.create_tfidf_batches(raw_corpus, dev)
            test = data_utils.read_annotations('../../askubuntu/test.txt', K_neg=-1, prune_pos_cnt=-1)
            test = data_utils.create_tfidf_batches(raw_corpus, test)
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
            embeddings, word_to_indx = data_utils.getGloveEmbeddingTensor(prune=True, cased=args.cased)
        else:
            embeddings, word_to_indx = data_utils.getEmbeddingTensor(args.embeddings_path)
        raw_corpus = data_utils.read_corpus(askubuntu_corpus)
        ids_corpus = data_utils.map_corpus(raw_corpus, word_to_indx, max_len=100)

        # model
        if args.snapshot is None:
            model = model_utils.get_model(embeddings, args, args.model_name)
            if args.gan_training:
                if args.simple_discriminator:
                    discriminator_name = 'simple_discriminator'
                else:
                    discriminator_name = 'discriminator'
                if args.complex_transformer:
                    transformer_name = 'complex_transformer'
                else:
                    transformer_name = 'transformer'
                transformer = model_utils.get_model(embeddings, args, transformer_name)
                discriminator = model_utils.get_model(embeddings, args, discriminator_name)
                encoder = model_utils.get_model(embeddings, args, 'encoder')
            elif args.domain_adaptation:
                model_2 = model_utils.get_model(None, args, args.model_name_2)
        else :
            print('\nLoading model from [%s]...' % args.snapshot)
            try:
                model = torch.load(args.snapshot)
                if args.domain_adaptation:
                    model_2 = torch.load(args.snapshot_2)
                # TODO add snapshot support for GAN training
            except :
                print("Sorry, This snapshot doesn't exist."); exit()
        print("Model:")
        print(model)
        if args.gan_training:
            print("Encoder Model:")
            print(encoder)
            print("Transformer Model:")
            print(transformer)
            print("Discriminator Model:")
            print(discriminator)
        elif args.domain_adaptation:
            print("Discriminator Model:")
            print(model_2)


        if args.android:
            raw_android_corpus = data_utils.read_corpus('../../Android/corpus.tsv.gz')
            ids_android_corpus = data_utils.map_corpus(raw_android_corpus, word_to_indx, max_len=100)
            dev_neg_dict = data_utils.read_annotations_android('../../Android/dev.neg.txt')
            dev_pos_dict = data_utils.read_annotations_android('../../Android/dev.pos.txt')
            test_neg_dict = data_utils.read_annotations_android('../../Android/test.neg.txt')
            test_pos_dict = data_utils.read_annotations_android('../../Android/test.pos.txt')
            dev = data_utils.create_eval_batches_android(ids_android_corpus, dev_pos_dict, dev_neg_dict)
            test = data_utils.create_eval_batches_android(ids_android_corpus, test_pos_dict, test_neg_dict)
        else:
            dev = data_utils.read_annotations('../../askubuntu/dev.txt', K_neg=-1, prune_pos_cnt=-1)
            dev = data_utils.create_eval_batches(ids_corpus, dev, 0, pad_left=False)
            test = data_utils.read_annotations('../../askubuntu/test.txt', K_neg=-1, prune_pos_cnt=-1)
            test = data_utils.create_eval_batches(ids_corpus, test, 0, pad_left=False)

        if args.train :
            train = data_utils.read_annotations('../../askubuntu/train_random.txt')
            # Create Batch2 batches
            if args.gan_training:
                encoder_batches = data_utils.create_batches(ids_corpus, train, args.batch_size, 0, pad_left=False)
                encoder_batches_vanilla = data_utils.create_batches(ids_corpus, train, args.batch_size, 0, pad_left=False)
                d_num_batches = (args.dk * len(train) / args.batch_size)
                if args.wgan:
                    d_num_batches += (100 * 25) + (100 * (len(train) / args.batch_size / 500))
                discriminator_batches = data_utils.create_discriminator_batches_parallel(ids_corpus, ids_android_corpus, d_num_batches, should_perm=False, pad_max=args.pad_max)
                transformer_batches = data_utils.create_discriminator_batches_parallel(ids_corpus, ids_android_corpus, (len(train) / args.batch_size), should_perm=False, pad_max=args.pad_max)
                train_utils.train_gan(encoder=encoder, transformer=transformer, discriminator=discriminator, encoder_batches=encoder_batches, encoder_batches_vanilla=encoder_batches_vanilla, discriminator_batches=discriminator_batches, transformer_batches=transformer_batches, dev_data=dev, test_data=test, args=args, embeddings=embeddings, results_lock=results_lock)
            elif args.domain_adaptation:
                train_2 = data_utils.create_discriminator_batches(ids_corpus, ids_android_corpus, (len(train) / args.batch_size + 1))
                train_utils.train_model(model, train, dev, test, ids_corpus, args.batch_size, args, model_2, train_2, results_lock)
            else:
                train_utils.train_model(model, train, dev, test, ids_corpus, args.batch_size, args)

        else:
            embedding_model = train_utils.get_embedding_model(embeddings, args.cuda)
            print("Evaluating performance on dev data...")
            MAP, MRR, P1, P5, auc5 = train_utils.evaluate(model_data=dev, model=model, args=args, embedding_model=embedding_model)
            print(tabulate([[MAP, MRR, P1, P5, auc5]], headers=['MAP', 'MRR', 'P@1', 'P@5', 'AUC0.05']))

            print()
            print("Evaluating performance on test data...")
            MAP, MRR, P1, P5, auc5 = train_utils.evaluate(model_data=test, model=model, args=args, embedding_model=embedding_model)
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
    parser.add_argument('-g', '--gan_training', action='store_true', default=False, help='choose gan training style')
    parser.add_argument('-w', '--wgan', action='store_true', default=False, help='wgan modifications on gan training style')
    parser.add_argument('-p', '--grad-penalty', action='store_true', default=False)
    # learning
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--lr2', type=float, default=-0.001, help='initial learning rate [default: -0.001]')
    parser.add_argument('--lr_d', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--lr_t', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs for train [default: 256]')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training [default: 64]')
    parser.add_argument('-k', '--dk', type=int, default=5, help='number of discriminator batches per transformer batch')
    parser.add_argument('--lam', type=float, default=0.001, help='constant multiplier on loss 2 in domain adaptation')
    parser.add_argument('--gam', type=float, default=0.1, help='constant multiplier on loss 2 in domain adaptation')
    # data
    parser.add_argument('--embeddings_path', type=str, default='../../askubuntu/vector/vectors_pruned.200.txt.gz', help='path for word embeddings')
    parser.add_argument('--cased', action='store_true', default=False, help="use cased glove embeddings")
    parser.add_argument('--pad_max', action='store_true', default=False, help="pad all sequences to max length of 100")
    # model
    parser.add_argument('--model_name', nargs="?", type=str, default='cnn3', choices=['dan', 'cnn2', 'cnn3', 'cnn4', 'lstm_bi', 'lstm_bi_fc', 'lstm3', 'tfidf'], help="Encoder model type [dan, cnn2, cnn3, cnn4, lstm_bi, lstm_bi_fc, lstm3]")
    parser.add_argument('--model_name_2', nargs="?", type=str, default='ffn', choices=['ffn'], help="Discriminator model type")
    parser.add_argument('--simple_discriminator', action='store_true', default=False, help="use simple linear discriminator")
    parser.add_argument('--complex_transformer', action='store_true', default=False, help="use more complex transformer")
    parser.add_argument('--num_hidden', type=int, default=512, help="encoding size.")
    parser.add_argument('--num_hidden_transformer', type=int, default=200, help="encoding size.")
    parser.add_argument('--num_hidden_discriminator', type=int, default=200, help="encoding size.")
    parser.add_argument('--dropout', type=float, default=0.0, help="dropout parameter")
    parser.add_argument('--dropout_d', type=float, default=0.0, help="dropout parameter for discriminator")
    parser.add_argument('--dropout_t', type=float, default=0.0, help="dropout parameter for transformer")
    parser.add_argument('--margin', type=float, default=1.0)
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
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="Verbose output")

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
