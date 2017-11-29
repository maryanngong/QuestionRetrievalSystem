import numpy as np
import torch
import torch.autograd as autograd
from sklearn.metrics.pairwise import cosine_similarity

# TODO convert to using PyTorch Varibales 
def batch_cosine_similarity(encoded, training_ids):
    encoded = encoded.data
    n = len(training_ids) # number of minibatches
    all_scores = []
    print("Cos Similarity")
    print(training_ids)
    for i in range(n):
        minibatch_indices = training_ids[i]
        enc_query = encoded[minibatch_indices[0]].numpy()
        enc_query = enc_query.reshape((1, -1))
        pos_id = minibatch_indices[1]
        negative_ids = minibatch_indices[2:]
        candidate_ids = minibatch_indices[1:]
        print(candidate_ids)

        enc_candidates = torch.index_select(encoded, 0, torch.LongTensor(candidate_ids)).numpy()
        similarity_scores = torch.from_numpy(cosine_similarity(enc_query, enc_candidates)).type(torch.FloatTensor) # shape should be (1, num_candidate_questions)
        all_scores.append(similarity_scores)
    return torch.stack(all_scores).view(n, -1), torch.zeros(n).type(torch.LongTensor)
