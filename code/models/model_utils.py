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
def get_model(embeddings, args, model_name):
    print("\nBuilding model...")

    if model_name == 'discriminator':
        return Discriminator(embeddings, args)
    elif model_name == 'transformer':
        return Transformer(embeddings, args)
    elif model_name == 'encoder':
        return CNN3(embeddings, args)
    elif model_name == 'ffn':
        return FFN(args)
    elif model_name == 'ffn2':
        return FFN2(args)
    elif model_name == 'cnn2':
        return CNN2(embeddings, args)
    elif model_name == 'cnn3':
        return CNN3(embeddings, args)
    elif model_name == 'lstm_bi':
        return LSTM_bi(embeddings, args)
    elif model_name == 'lstm_bi_fc':
        return LSTM_bi_fc(embeddings, args)
    elif model_name == 'lstm3':
        return LSTM3(embeddings, args)
    else:
        raise Exception("Model name {} not supported!".format(args.model_name))


class Transformer(nn.Module):

    def __init__(self, embeddings, args):
        super(Transformer, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape

        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer.weight.requires_grad = False

        self.fc1 = nn.Linear(embed_dim + 1, args.num_hidden_transformer) # Extra channel for noise
        self.fc2 = nn.Linear(args.num_hidden_transformer, args.num_hidden_transformer)
        self.fc3 = nn.Linear(args.num_hidden_transformer, embed_dim)

    def forward(self, x):
        x = self.embedding_layer(x)
        noise = torch.autograd.Variable(torch.FloatTensor(x.size()[0], x.size()[1], 1).uniform_(-1, 1))
        if self.args.cuda:
            noise = noise.cuda()
        torch.cat((x, noise), 2)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=args.dropout_t, training=self.training)
        x = F.sigmoid(self.fc2(x))
        out = self.fc3(x)
        return out


class Discriminator(nn.Module):

    def __init__(self, embeddings, args):
        super(Discriminator, self).__init__()
        self.args = args
        _, embed_dim = embeddings.shape
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=args.num_hidden_discriminator // 2,
                          num_layers=1, batch_first=True, bidirectional=True, dropout=args.dropout_d)
        self.fc1 = nn.Linear(args.num_hidden_discriminator, 1)

    def init_hidden_states(self, batch_size):
        h0 = autograd.Variable(torch.randn(2, batch_size, self.args.num_hidden_discriminator // 2))
        c0 = autograd.Variable(torch.randn(2, batch_size, self.args.num_hidden_discriminator // 2))
        if self.args.cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)

    def forward(self, x):
        batch_size = x.size()[0]
        h0, c0 = self.init_hidden_states(batch_size)
        _, (h_n, c_n) = self.rnn(x, (h0, c0))
        out = self.fc1(h_n)
        return out


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


    def forward(self, x_indx):
        x = self.embedding_layer(x_indx)
        # reorder dimensions for convolutional layer
        x = x.permute(0,2,1)
        x = F.dropout(x, p=self.args.dropout, training=self.training)
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
        # self.fc = nn.Linear(args.num_hidden, args.num_hidden)


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
        # output = self.fc(output)
        # print("output size", output.size())
        return output

class LSTM_bi_fc(nn.Module):

    def __init__(self, embeddings, args):
        super(LSTM_bi_fc, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape
        self.embed_dim = embed_dim
        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer.weight.requires_grad = False
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=args.num_hidden // 2,
                          num_layers=1, batch_first=True, bidirectional=True, dropout=args.dropout)
        self.fc = nn.Linear(args.num_hidden, args.num_hidden)
        self.tanh = nn.Tanh()


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
        output = self.tanh(self.fc(output))
        return output

class FFN(nn.Module):

    def __init__(self, args):
        super(FFN, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.num_hidden, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x_indx):
        hidden = self.fc1(x_indx)
        out = self.fc2(hidden)
        return out

class FFN2(nn.Module):

    def __init__(self, args):
        super(FFN2, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.num_hidden, 300)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(300, 150)
        self.fc3 = nn.Linear(150, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x_indx):
        hidden = self.fc1(x_indx)
        out = self.fc2(hidden)
        out = self.relu(out)
        out = self.fc3(out)
        return self.sig(out)
