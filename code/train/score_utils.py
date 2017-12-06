import numpy as np
import torch
import torch.autograd as autograd
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

def batch_cosine_similarity(encoded, training_ids, cuda=False):
    n = len(training_ids) # number of minibatches
    all_scores = []
    for i in range(n):
        minibatch_indices = training_ids[i]
        enc_query = encoded[minibatch_indices[0]]
        pos_id = minibatch_indices[1]
        negative_ids = minibatch_indices[2:]
        candidate_ids = minibatch_indices[1:]

        select_indices = autograd.Variable(torch.LongTensor(candidate_ids))
        if cuda:
            select_indices = select_indices.cuda()
        enc_candidates = encoded.index_select(0, select_indices)
        similarity_scores = F.cosine_similarity(enc_query.view(1,-1), enc_candidates)
        all_scores.append(similarity_scores)

    target_indices = autograd.Variable(torch.zeros(n).type(torch.LongTensor))
    if cuda:
        target_indices = target_indices.cuda()
    return torch.cat(all_scores, 0).view(n, -1), target_indices 


def batch_cosine_similarity_eval(encoded, cuda=False):
    enc_query = encoded[0]
    enc_candidates = encoded[1:]
    similarity_scores = F.cosine_similarity(enc_query.view(1,-1), enc_candidates)
    return similarity_scores
