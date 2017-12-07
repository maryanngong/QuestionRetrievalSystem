import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import tqdm
import datetime
import pdb


# Depending on arg, build dataset
def get_model(embeddings, args):
    print("\nBuilding model...")

    if args.model_name == 'dan':
        return DAN(embeddings, args)
    elif args.model_name == 'cnn2':
        return CNN2(embeddings, args)
    elif args.model_name == 'cnn3':
        return CNN3(embeddings, args)
    elif args.model_name == 'cnn4':
        return CNN4(embeddings, args)
    elif args.model_name == 'lstm_bi':
        return LSTM_bi(embeddings, args)
    elif args.model_name == 'lstm3':
        return LSTM3(embeddings, args)
    else:
        raise Exception("Model name {} not supported!".format(args.model_name))


class DAN(nn.Module):

    def __init__(self, embeddings, args):
        super(DAN, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape
        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.W_hidden = nn.Linear(embed_dim, 32) # we should make this 32 an argument, which is final encoding size
        # self.W_out = nn.Linear(32, 1)

    def forward(self, x_indx):
        all_x = self.embedding_layer(x_indx)
        avg_x = torch.mean(all_x, dim=1)
        hidden = F.tanh( self.W_hidden(avg_x) )
        # out = self.W_out(hidden)
        return hidden


class CNN2(nn.Module):

    def __init__(self, embeddings, args):
        super(CNN2, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape

        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )

        self.conv1 = nn.Conv1d(embed_dim, args.num_hidden, kernel_size=3)


    def forward(self, x_indx):
        x = self.embedding_layer(x_indx)
        x =x.permute(0,2,1)
        out = self.conv1(x)
        out = torch.mean(out, 2)
        # print("size of out", out.size())
        return out

class CNN3(nn.Module):

    def __init__(self, embeddings, args):
        super(CNN3, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape

        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer.weight.requires_grad = False

        self.conv1 = nn.Conv1d(embed_dim, args.num_hidden, kernel_size=3)
        # self.dropout = nn.Dropout(p=args.dropout)


    def forward(self, x_indx):
        x = self.embedding_layer(x_indx)
        # reorder dimensions for convolutional layer
        x =x.permute(0,2,1)
        # x = self.dropout(x)
        # Trying functional dropout (bc it doesnt need to be trained) with training flag set properly
        x = F.dropout(x, p=args.dropout, training=self.training)
        out = self.conv1(x)
        return out

class CNN4(nn.Module):

    def __init__(self, embeddings, args):
        super(CNN4, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape

        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer.weight.requires_grad = False
        self.conv1 = nn.Conv1d(embed_dim, args.num_hidden, kernel_size=3)
        # self.fc1 = nn.Linear(args.num_hidden * 3, args.num_hidden)

    def forward(self, x_indx):
        x = self.embedding_layer(x_indx)
        # reorder dimensions for convolutional layer
        x =x.permute(0,2,1)
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        out = self.conv1(x)
        out = F.tanh(out)
        return out


class LSTM3(nn.Module):

    def __init__(self, embeddings, args):
        super(LSTM3, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape
        self.embed_dim = embed_dim
        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=args.num_hidden,
                          num_layers=1, batch_first=True, dropout=args.dropout)


    def init_hidden_states(self, batch_size):
        h0 = autograd.Variable(torch.randn(1, batch_size, self.args.num_hidden))
        c0 = autograd.Variable(torch.randn(1, batch_size, self.args.num_hidden))
        if self.args.cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)


    def forward(self, x_indx):
        all_x = self.embedding_layer(x_indx)
        batch_size = len(x_indx)
        h0, c0 = self.init_hidden_states(batch_size)
        output, (h_n, c_n) = self.rnn(all_x, (h0, c0))
        return output

class LSTM_bi(nn.Module):

    def __init__(self, embeddings, args):
        super(LSTM_bi, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape
        self.embed_dim = embed_dim
        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer.weight.requires_grad = False
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=args.num_hidden // 2,
                          num_layers=1, batch_first=True, bidirectional=True, dropout=args.dropout)


    def init_hidden_states(self, batch_size):
        h0 = autograd.Variable(torch.randn(2, batch_size, self.args.num_hidden // 2))
        c0 = autograd.Variable(torch.randn(2, batch_size, self.args.num_hidden // 2))
        if self.args.cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)


    def forward(self, x_indx):
        all_x = self.embedding_layer(x_indx)
        batch_size = len(x_indx)
        h0, c0 = self.init_hidden_states(batch_size)
        output, (h_n, c_n) = self.rnn(all_x, (h0, c0))
        return output
