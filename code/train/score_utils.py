import numpy as np
import torch
import torch.autograd as autograd
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F


# Calculates cosine similarity for this batch of data
# Parameters:
#   - encoded: whole batch output of network model  Nxhidden_size
#   - training_ids: training group ids, organized as a list of minibatches
#                   - each minibatch is a list of indices into the encoded matrix
#                   - the first index refers to the query question
#                   - second index is the positive question (this is only for training batches)
#                   - rest of the indices are the negative questions
#   - cuda: boolean flag
# Returns:
#   - scores: these are the cosine similarities computed between all candidate questions and each query question for every minibatch  len(training_ids)xnum_candidates
#   - target_indices: these are the target y's to be passed to the MultiMarginLoss fxn (size len(training_ids)x1 )
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


# Calculates cosine similarity for this batch of data, used during evaluation
# Parameters:
#   - encoded: whole batch output of network model  Nxhidden_size
#               - NOTE: we assume that the query question is the first row of encoded, all the rest are candidate question encodings
#   - cuda: boolean flag
# Returns:
#   - scores: these are the cosine similarities computed between the query question and all candidates (output size: num_candidate questions)
def batch_cosine_similarity_eval(encoded, cuda=False):
    enc_query = encoded[0]
    enc_candidates = encoded[1:]
    similarity_scores = F.cosine_similarity(enc_query.view(1,-1), enc_candidates)
    return similarity_scores
