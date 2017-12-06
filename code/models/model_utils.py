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
    elif args.model_name == 'cnn':
        return CNN(embeddings, args)
    elif args.model_name == 'cnn2':
        return CNN2(embeddings, args)
    elif args.model_name == 'cnn3':
        return CNN3(embeddings, args)
    elif args.model_name == 'rnn':
        return RNN(embeddings, args)
    elif args.model_name == 'lstm':
        return LSTM(embeddings, args)
    elif args.model_name == 'lstm2' or 'lstm_bi':
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
        # self.pool1 = nn.AvgPool1d()

        self.layer1 = nn.Sequential(
            nn.Conv1d(embed_dim, 32, kernel_size=3),
            # nn.BatchNorm1d(16),
            # nn.ReLU(),
            nn.AvgPool1d(2))
        self.fc = nn.Linear(32*15, args.num_hidden)

    # def forward(self, x_indx):
    #     x = self.embedding_layer(x_indx)
    #     x = x.permute(0,2,1)
    #     print("size x after embedding", x.size())
    #     out = self.layer1(x)
    #     print("size out", out.size())
    #     out = out.view(out.size(0), -1)
    #     print("view out", out.size())
    #     out = self.fc(out)
    #     print("after fc out", out.size())
    #     return out

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

        self.conv1 = nn.Conv1d(embed_dim, args.num_hidden, kernel_size=3)
        # self.pool1 = nn.AvgPool1d()



    def forward(self, x_indx):
        x = self.embedding_layer(x_indx)
        x =x.permute(0,2,1)
        out = self.conv1(x)
        return out


# TODO Finish modifying DAN code to use CNN-style layers
class CNN(nn.Module):

    def __init__(self, embeddings, args):
        super(CNN, self).__init__()
        self.args = args
        num_channels_in = 1
        num_channels_out = 5 # this should probably be an arg that we specify
        vocab_size, embed_dim = embeddings.shape
        kernel_sizes = [(3, embed_dim), (4, embed_dim), (5, embed_dim)]

        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )

        self.conv1 = nn.Conv2d(1, num_channels_out, kernel_sizes[0])
        # self.conv2 = nn.Conv2d(1, num_channels_out, kernel_sizes[1])
        self.dropout = nn.Dropout(0.5)
        # the 2 refers to having 2 conv layers
        self.fc1 = nn.Linear(1*num_channels_out, args.num_hidden)

    def conv_and_pool(self, x, conv, max_pool=False):
        x = F.relu(conv(x)).squeeze(3) #(N, num_channels_out, W)
        if max_pool:
            x = F.max_pool1d(x, x.size(2)).squeeze(2)
        else: # mean pool instead
            x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x_indx):
        x = self.embedding_layer(x_indx)
        x = x.unsqueeze(1)
        x1 = self.conv_and_pool(x, self.conv1)
        # x2 = self.conv_and_pool(x, self.conv2)

        # x = torch.cat([x1, x2], 1)
        x = x1
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit # (N,C)


class RNN(nn.Module):

    def __init__(self, embeddings, args):
        super(RNN, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape
        self.embed_dim = embed_dim
        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=200,
                          num_layers=1, batch_first=True)
        self.W_o = nn.Linear(200,1)

    def forward(self, x_indx):
        all_x = self.embedding_layer(x_indx)
        h0 = autograd.Variable(torch.randn(1, self.args.batch_size, 200))
        output, h_n = self.rnn(all_x, h0)
        h_n = h_n.squeeze(0)
        out = self.W_o(h_n )
        return out


class LSTM(nn.Module):

    def __init__(self, embeddings, args):
        super(LSTM, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape
        self.embed_dim = embed_dim
        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=args.num_hidden,
                          num_layers=1, batch_first=True)
        # self.W_o = nn.Linear(args.num_hidden,1)

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
        # print("shape of output", output.size())
        output = torch.mean(output, 1)
        return output
        # out = self.W_o(h_n )
        # return out

class LSTM3(nn.Module):

    def __init__(self, embeddings, args):
        super(LSTM3, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape
        self.embed_dim = embed_dim
        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=args.num_hidden,
                          num_layers=1, batch_first=True)
        # self.W_o = nn.Linear(args.num_hidden,1)

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
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=args.num_hidden // 2,
                          num_layers=1, batch_first=True, bidirectional=True)
        # self.W_o = nn.Linear(args.num_hidden,1)

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
