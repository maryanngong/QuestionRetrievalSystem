import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

def batch_cosine_similarity(encoded, training_ids):
    n = len(training_ids) # number of minibatches
    all_scores = []
    for i in range(n):
        minibatch_indices = training_ids[i]
        enc_query = encoded[minibatch_indices[0]] # or should I use [j,:]? 2D?
        pos_id = minibatch_indices[1]
        negative_ids = minibatch_indices[2:]
        candidate_ids = minibatch_indices[1:]

        enc_candidates = encoded[candidate_ids]
        similarity_scores = torch.from_numpy(cosine_similarity(enc_query, enc_candidates)).type(torch.FloatTensor) # shape should be (1, num_candidate_questions)
        all_scores.append(similarity_scores)
    return torch.stack(all_scores), torch.zeros(n).type(torch.LongTensor)
