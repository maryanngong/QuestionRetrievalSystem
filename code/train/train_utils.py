import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
import datetime
import pdb
import numpy as np

def train_model(train_data, dev_data, model, args):
    if args.cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters() , lr=args.lr)
    model.train()

    for epoch in range(1, args.epochs+1):
        print("-------------\nEpoch {}:\n".format(epoch))

        loss = run_epoch(train_data, True, model, optimizer, args)
        print('Train max-margin loss: {:.6f}'.format( loss))
        print()

        val_loss = run_epoch(dev_data, False, model, optimizer, args)
        print('Val max-margin loss: {:.6f}'.format( val_loss))

        # Save model
        torch.save(model, args.save_path)

def run_epoch(data_batches, is_training, model, optimizer, args):
    '''
    Train model for one pass of train data, and return loss, acccuracy
    '''
    losses = []
    if is_training:
        model.train()
    else:
        model.eval()
    N = len(data)
    for i in xrange(N):
        t, b, g = data_batches[i]
        # Titles, Bodies are text samples (tokenized words are already converted to indices for embedding layer)
        # Train Group IDs are the IDs of data samples where each sample is (query, positive examples, negative examples)
        titles, bodies, train_group_ids = autograd.Variable(t), autograd.Variable(b), autograd.Variable(g)
        if args.cuda:
            titles, bodies, train_group_ids = titles.cuda(), bodies.cuda(), train_group_ids.cuda()
        if is_training:
            optimizer.zero_grad()
        # Encode all of the title and body text using model
        text_encodings = model(titles, bodies)
        # Calculate Loss = Multi-Margin-Loss(train_group_ids, text_encodings)
        scores, target_indices = cosine_differences(train_group_ids, text_encodings)
        loss = F.multi_margin_loss(scores, target_indices)
        if is_training:
            loss.backward()
            optimizer.step()
        losses.append(loss.cpu().data[0])
    avg_loss = np.mean(losses)
    return avg_loss
