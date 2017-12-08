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
from itertools import ifilter
from evaluation import Evaluation
import data.myio as myio
from meter import AUCMeter


# Takes in raw dataset and masks out padding and then takes sum average for LSTM
def average_without_padding_lstm(x, ids, eps=1e-8, padding_id=0, cuda=True):
    #normalize x
    x = F.normalize(x,dim=2)
    mask = ids != 0
    mask = mask.type(torch.FloatTensor)
    mask = autograd.Variable(mask.unsqueeze(2), requires_grad=False)
    if cuda:
        mask = mask.cuda()

    s = torch.sum(x*mask, 1) / (torch.sum(mask, 1)+eps)
    return s

# Takes in raw dataset and masks out padding and then takes sum average for CNN
def average_without_padding_cnn(x, ids, eps=1e-8, padding_id=0, cuda=True):
    # normalize x
    x = F.normalize(x, dim=1)
    mask = ids != 0
    mask = mask[:,:-2]
    mask = mask.type(torch.FloatTensor)
    mask = autograd.Variable(mask.unsqueeze(1), requires_grad=False)
    if cuda:
        mask = mask.cuda()

    s = torch.sum(x*mask, 2) / (torch.sum(mask, 2)+eps)
    return s

def get_scores(x):
    print("shape of x", x.size())
    x = F.normalize(x, dim=1)
    scores = torch.mm(x[1:], x[0].unsqueeze(1))
    print("shape of scores", scores.size())
    return scores

def get_scores_train(h_final, idps, n_d, args):
    # print("hfinal shape", h_final.size())
    indices = torch.LongTensor(idps.ravel())
    if args.cuda:
        indices = indices.cuda()
    xp = h_final[indices]
    n = idps.shape[0]
    xp = xp.view(idps.shape[0], idps.shape[1], n_d)
    # num query * n_d
    query_vecs = xp[:,0,:].unsqueeze(1)
    scores = torch.sum(query_vecs*xp[:,1:,:], 2)
    target_indices = autograd.Variable(torch.zeros(n).type(torch.LongTensor))
    if args.cuda:
        target_indices = target_indices.cuda()
    return scores, target_indices

def evaluate(data, model, args):
    res = [ ]
    meter = AUCMeter()
    for idts, idbs, labels in data:
        titles, bodies = autograd.Variable(idts), autograd.Variable(idbs)
        if args.cuda:
            titles, bodies = titles.cuda(), bodies.cuda()
        encode_titles = model(titles)
        encode_bodies = model(bodies)

        if 'lstm' in args.model_name:
            titles_encodings = average_without_padding_lstm(encode_titles, idts)
            bodies_encodings = average_without_padding_lstm(encode_bodies, idbs)
        else:
            titles_encodings = average_without_padding_cnn(encode_titles, idts)
            bodies_encodings = average_without_padding_cnn(encode_bodies, idbs)

        text_encodings = (titles_encodings + bodies_encodings) * 0.5
        scores = score_utils.batch_cosine_similarity_eval(text_encodings, cuda=args.cuda).data.cpu().numpy()
        meter.add(scores, labels)
        assert len(scores) == len(labels)
        ranks = (-scores).argsort()
        ranked_labels = labels[ranks]
        res.append(ranked_labels)
    e = Evaluation(res)
    MAP = e.MAP()*100
    MRR = e.MRR()*100
    P1 = e.Precision(1)*100
    P5 = e.Precision(5)*100
    auc5 = meter.value(max_fpr=0.05)
    return MAP, MRR, P1, P5, auc5

def train_model(model, train, dev_data, test_data, ids_corpus, batch_size, args):
    is_training=True
    if args.cuda:
        model = model.cuda()
    parameters = ifilter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters , lr=args.lr)
    model.train()

    best_epoch = 0
    best_MRR = 0
    for epoch in range(1, args.epochs+1):
        print("-------------\nEpoch {}:\n".format(epoch))
        # randomize new dataset each time
        train_batches = myio.create_batches(ids_corpus, train, batch_size, 0, pad_left=False)
        
        N =len(train_batches)
        losses = []
        all_results = []
        for i in xrange(N):
            # get current batch
            t, b, train_group_ids = train_batches[i]
            titles, bodies = autograd.Variable(t), autograd.Variable(b)
            if args.cuda:
                titles, bodies = titles.cuda(), bodies.cuda()
            if is_training:
                optimizer.zero_grad()
            # Encode all of the title and body text using model
            # squeeze dimension differs for lstm and cnn models

            encode_titles = model(titles)
            encode_bodies = model(bodies)

            if 'cnn' in args.model_name:
                titles_encodings = average_without_padding_cnn(encode_titles, t)
                bodies_encodings = average_without_padding_cnn(encode_bodies, b)
            else:
                titles_encodings = average_without_padding_lstm(encode_titles, t)
                bodies_encodings = average_without_padding_lstm(encode_bodies, b)

            text_encodings = (titles_encodings + bodies_encodings) * 0.5

            # Calculate Loss = Multi-Margin-Loss(train_group_ids, text_encodings)
            scores, target_indices = score_utils.batch_cosine_similarity(text_encodings, train_group_ids, args.cuda)

            loss = F.multi_margin_loss(scores, target_indices, margin=args.margin)
            if is_training:
                loss.backward()
                optimizer.step()
            losses.append(loss.cpu().data[0])
            print("BATCH LOSS "+str(i+1)+" out of "+str(N)+": ")
            print(loss.cpu().data[0])
            rankings = compile_rankings(train_group_ids, scores.cpu().data.numpy())
            results = strip_ids_and_scores(rankings)
            all_results += results

        # if epoch % 2 == 0 and len(args.save_path) > 0:
        #     torch.save(model, args.save_path+"_size"+str(args.num_hidden)+"_epoch"+str(epoch))
        # Evaluation Metrics
        precision_at_1 = eval_utils.precision_at_k(all_results, 1)*100
        precision_at_5 = eval_utils.precision_at_k(all_results, 5)*100
        MAP = eval_utils.mean_average_precision(all_results)*100
        MRR = eval_utils.mean_reciprocal_rank(all_results)*100
        print(tabulate([[MAP, MRR, precision_at_1, precision_at_5]], headers=['MAP', 'MRR', 'P@1', 'P@5']))
        avg_loss = np.mean(losses)
        print('Train max-margin loss: {:.6f}'.format( avg_loss))

        print("Dev data Performance")
        MAP, MRR, P1, P5, auc5 = evaluate(dev_data, model, args)
        print(tabulate([[MAP, MRR, P1, P5, auc5]], headers=['MAP', 'MRR', 'P@1', 'P@5', 'AUC0.05']))
        print()
        print("Test data Performance")
        mapt, mrrt, p1t, p5t, auc5t = evaluate(test_data, model, args)
        print(tabulate([[mapt, mrrt, p1t, p5t, auc5t]], headers=['MAP', 'MRR', 'P@1', 'P@5', 'AUC0.05']))
        print()

        # for android dataset, validate using AUC0.05 metric
        if args.android:
            MRR = auc5
        if MRR > best_MRR:
            best_MRR = MRR
            best_epoch = epoch
            # Save model
            torch.save(model, args.save_path+"_size"+str(args.num_hidden)+"_epoch_best")


        print("Best EPOCH so far:", best_epoch, best_MRR)
        print()

# same as compile_rankings function except it already has positive and negative labels passed as qlabels
# also only handles one minibatch
def compile_rankings_eval(qlabels, scores):
    all_rankings = []
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
