import numpy as np
import torch
import torch.autograd as autograd
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

def batch_cosine_similarity(encoded, training_ids, cuda=False):
    # encoded = encoded.data
    n = len(training_ids) # number of minibatches
    # print("type", type(training_ids), type(training_ids[0]))
    all_scores = []
    # print("Cos Similarity")
    for i in range(n):
        minibatch_indices = training_ids[i]
        enc_query = encoded[minibatch_indices[0]]
        # enc_query = enc_query.reshape((1, -1))
        pos_id = minibatch_indices[1]
        negative_ids = minibatch_indices[2:]
        candidate_ids = minibatch_indices[1:]
        # print("candidate_ids", candidate_ids)
        # print("max candidate_id", max(candidate_ids))
        # print("shape of encoded", encoded.data.shape)

        select_indices = autograd.Variable(torch.LongTensor(candidate_ids))
        if cuda:
            select_indices = select_indices.cuda()
        enc_candidates = encoded.index_select(0, select_indices)
        similarity_scores = F.cosine_similarity(enc_query.view(1,-1), enc_candidates)
        # similarity_scores = torch.from_numpy(cosine_similarity(enc_query, enc_candidates)).type(torch.FloatTensor) # shape should be (1, num_candidate_questions)
        all_scores.append(similarity_scores)

    target_indices = autograd.Variable(torch.zeros(n).type(torch.LongTensor))
    if cuda:
        target_indices = target_indices.cuda()
    return torch.cat(all_scores, 0).view(n, -1), target_indices 


def batch_cosine_similarity_eval(encoded, cuda=False):
    # print("shape of encoded", encoded.size())
    enc_query = encoded[0]
    enc_candidates = encoded[1:]
    # print("num enc_cnadidates", len(enc_candidates))
    similarity_scores = F.cosine_similarity(enc_query.view(1,-1), enc_candidates)
    # print("shape of similarlity scores", similarity_scores.size())
    return similarity_scores
