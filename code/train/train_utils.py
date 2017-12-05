import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from tabulate import tabulate
import datetime
import pdb
import numpy as np
import score_utils as score_utils
import evaluation_utils as eval_utils

def train_model(dataset, dev_data, model, args):
    if args.cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters() , lr=args.lr)
    model.train()

    best_epoch = 0
    best_MAP = 0
    for epoch in range(1, args.epochs+1):
        print("-------------\nEpoch {}:\n".format(epoch))
        # randomize new dataset each time
        train_data = dataset.get_train_batches()
        loss = run_epoch(train_data, True, model, optimizer, args)
        print('Train max-margin loss: {:.6f}'.format( loss))
        print()

        if epoch%2 == 1:
            # Save model
            torch.save(model, args.save_path+"_size"+str(args.num_hidden)+"_epoch"+str(epoch))

        # val_loss = run_epoch(dev_data, False, model, optimizer, args)
        # print('Val max-margin loss: {:.6f}'.format( val_loss))
        MAP, MRR = eval_model_two(dev_data, model, args)
        if MAP > best_MAP:
            best_MAP = MAP
            best_epoch = epoch
        print("Best EPOCH so far:", best_epoch, best_MAP)

    # save final model
    # torch.save(model, args.save_path+"_size"+str(args.num_hidden)+"_final")
    print("Best EPOCH:", best_epoch, best_MAP)

def eval_model_two(data_batches, model, args):
    if args.cuda:
        model = model.cuda()

    model.eval()
    N = len(data_batches)
    all_results = []
    for i in xrange(N):
        t, b, qlabels = data_batches[i]
        assert qlabels[0] == -1
        # print("lengths in eval", t.shape, b.shape, len(qlabels))
        titles, bodies = autograd.Variable(t), autograd.Variable(b)
        if args.cuda:
            titles, bodies = titles.cuda(), bodies.cuda()
        titles_encodings = F.normalize(model(titles))
        bodies_encodings = F.normalize(model(bodies))
        text_encodings = (titles_encodings + bodies_encodings) * 0.5
        # Calculate Loss = Multi-Margin-Loss(train_group_ids, text_encodings)
        scores = score_utils.batch_cosine_similarity_eval(text_encodings, cuda=args.cuda)
        rankings = compile_rankings_eval(qlabels[1:], scores.cpu().data.numpy())
        results = strip_ids_and_scores([rankings])
        all_results += results
    # print("all results", all_results)
    precision_at_1 = eval_utils.precision_at_k(all_results, 1)
    precision_at_5 = eval_utils.precision_at_k(all_results, 5)
    MAP = eval_utils.mean_average_precision(all_results)
    MRR = eval_utils.mean_reciprocal_rank(all_results)
    print(tabulate([[MAP, MRR, precision_at_1, precision_at_5]], headers=['MAP', 'MRR', 'P@1', 'P@5']))   
    return MAP, MRR 

def eval_model(data_batches, model, args):
    if args.cuda:
        model = model.cuda()

    model.eval()
    N = len(data_batches)
    all_results = []
    for i in xrange(N):
        t, b, g = data_batches[i]
        train_group_ids = g
        # Titles, Bodies are text samples (tokenized words are already converted to indices for embedding layer)
        # Train Group IDs are the IDs of data samples where each sample is (query, positive examples, negative examples)
        titles, bodies = autograd.Variable(t), autograd.Variable(b)
        if args.cuda:
            titles, bodies = titles.cuda(), bodies.cuda() #, train_group_ids.cuda() <-- i don't think this needs to be a cuda variable

        # Encode all of the title and body text using model

        titles_encodings = F.normalize(model(titles))
        bodies_encodings = F.normalize(model(bodies))
        text_encodings = (titles_encodings + bodies_encodings) * 0.5
        # Calculate Loss = Multi-Margin-Loss(train_group_ids, text_encodings)
        scores, target_indices = score_utils.batch_cosine_similarity(text_encodings, train_group_ids, args.cuda)
        # loss = F.multi_margin_loss(scores, target_indices)
        # print("loss:", loss)
        # Evaluation Metrics
        rankings = compile_rankings(train_group_ids, scores.cpu().data.numpy())
        results = strip_ids_and_scores(rankings)
        all_results += results

    precision_at_1 = eval_utils.precision_at_k(all_results, 1)
    precision_at_5 = eval_utils.precision_at_k(all_results, 5)
    MAP = eval_utils.mean_average_precision(all_results)
    MRR = eval_utils.mean_reciprocal_rank(all_results)
    print(tabulate([[MAP, MRR, precision_at_1, precision_at_5]], headers=['MAP', 'MRR', 'P@1', 'P@5']))
    return MAP, MRR


def run_epoch(data_batches, is_training, model, optimizer, args):
    '''
    Train model for one pass of train data, and return loss, acccuracy
    '''
    losses = []
    if is_training:
        model.train()
    else:
        model.eval()
    N = len(data_batches)
    all_train_group_ids = None
    all_scores = None
    all_results = []
    for i in xrange(N):
        t, b, g = data_batches[i]
        train_group_ids = g
        # Titles, Bodies are text samples (tokenized words are already converted to indices for embedding layer)
        # Train Group IDs are the IDs of data samples where each sample is (query, positive examples, negative examples)
        titles, bodies = autograd.Variable(t), autograd.Variable(b)
        if args.cuda:
            titles, bodies = titles.cuda(), bodies.cuda() #, train_group_ids.cuda() <-- i don't think this needs to be a cuda variable
        if is_training:
            optimizer.zero_grad()
        # Encode all of the title and body text using model
        titles_encodings = F.normalize(model(titles))
        bodies_encodings = F.normalize(model(bodies))
        print("sizes of encodings", type(titles_encodings), titles_encodings.size(), bodies_encodings.size())
        text_encodings = (titles_encodings + bodies_encodings) * 0.5
        # Calculate Loss = Multi-Margin-Loss(train_group_ids, text_encodings)
        scores, target_indices = score_utils.batch_cosine_similarity(text_encodings, train_group_ids, args.cuda)
        # print('SHAPE SCORES:')
        # print(scores.data.shape)
        # print('SHAPE TARGET_INDICES:')
        # print(target_indices.data.shape)
        loss = F.multi_margin_loss(scores, target_indices)
        if is_training:
            loss.backward()
            optimizer.step()
        losses.append(loss.cpu().data[0])
        print("BATCH LOSS "+str(i+1)+" out of "+str(N)+": ")
        print(loss.cpu().data[0])
        rankings = compile_rankings(train_group_ids, scores.cpu().data.numpy())
        results = strip_ids_and_scores(rankings)
        all_results += results
        # Concat with cumulative vars for eval at end of epoch
        # if all_train_group_ids is None:
        #     all_train_group_ids = train_group_ids
        # else:
        #     all_train_group_ids = np.vstack((all_train_group_ids, train_group_ids))
        # if all_scores is None:
        #     all_scores = scores
        # else:
        #     all_scores = torch.cat((all_scores, scores))
    # Evaluation Metrics
    precision_at_1 = eval_utils.precision_at_k(all_results, 1)
    precision_at_5 = eval_utils.precision_at_k(all_results, 5)
    MAP = eval_utils.mean_average_precision(all_results)
    MRR = eval_utils.mean_reciprocal_rank(all_results)
    print(tabulate([[MAP, MRR, precision_at_1, precision_at_5]], headers=['MAP', 'MRR', 'P@1', 'P@5']))
    # rankings = compile_rankings(all_train_group_ids, all_scores.cpu().data.numpy())
    # results = strip_ids_and_scores(rankings)
    # precision_at_1 = eval_utils.precision_at_k(results, 1)
    # precision_at_5 = eval_utils.precision_at_k(results, 5)
    # MAP = eval_utils.mean_average_precision(results)
    # MRR = eval_utils.mean_reciprocal_rank(results)
    # print(tabulate([[MAP, MRR, precision_at_1, precision_at_5]], headers=['MAP', 'MRR', 'P@1', 'P@5']))
    avg_loss = np.mean(losses)
    return avg_loss

def compile_rankings_eval(qlabels, scores):
    all_rankings = []
    # print("sum of qlabels", sum(qlabels))
    # print("shape of scores", scores.shape)
    for i in range(len(qlabels)):
        entry = (None, scores[i], qlabels[i])
        all_rankings.append(entry)
    return sorted(all_rankings, key=lambda x: x[1], reverse=True)

def compile_rankings(train_group_ids, scores_variable):
    '''Compiles a list of lists ranking the queries by their scores

    Args:
        train_group_ids
        scores

    Returns:
        rankings list(list(tuple(id, score, relevant)))

    >>> compile_rankings([[123, 200, 567, 876, 987], [123, 201, 567, 876, 987], [134, 243, 902, 581, 939]], [[0.91, 0.1, 0.25, 0.3], [0.97, 0.1, 0.25, 0.3], [0.84, 0.41, 0.15, 0.2001]])
    [[(201, 0.97, 1), (200, 0.91, 1), (987, 0.3, 0), (876, 0.25, 0), (567, 0.1, 0)], [(243, 0.84, 1), (902, 0.41, 0), (939, 0.2001, 0), (581, 0.15, 0)]]

    '''
    scores = scores_variable #.data
    all_rankings, rankings = [], []
    last_question_id = None
    for i, group in enumerate(train_group_ids):
        if group[0] != last_question_id:
            if len(rankings) > 0:
                all_rankings.append(sorted(rankings, key=lambda q: q[1], reverse=True))
            rankings = []
            last_question_id = group[0]
            # Append negative queries just once
            for j, negative_id in enumerate(group[2:]):
                # index = group_index - 1 = j + 2 - 1 = j + 1
                entry = (negative_id, scores[i][j+1], 0)
                rankings.append(entry)
        # Append this entry's positive query
        entry = (group[1], scores[i][0], 1)
        rankings.append(entry)
    all_rankings.append(sorted(rankings, key=lambda q: q[1], reverse=True))
    return all_rankings

def strip_ids_and_scores(rankings):
    '''Strips the ids and scores from a list of rankings

    Args:
        rankings: (see compile_rankings for description)

    Returns:
        list of lists or relevance values: 1 if that query was relevant, 0 otherwise

    >>> strip_ids_and_scores([[(243, 0.8, 1), (989, 0.45, 0), (544, 0.47, 1)],[(12, 0.99, 0), (912, 0.34, 1), (453, 0.6532, 1)],[(888, 0.3, 0), (97, 0.1, 0), (334, 0.2, 0)]])
    [[1, 0, 1], [0, 1, 1], [0, 0, 0]]

    '''
    results = []
    for ranking in rankings:
        result = [entry[2] for entry in ranking]
        results.append(result)
    return results

if __name__ == '__main__':
    import doctest
    doctest.testmod()
