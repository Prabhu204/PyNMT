# -*- coding: utf-8 -*-
# @Author : Prabhu Appalapuri<prabhu.appalapuri@gmail.com>
# @Time : 20.02.20 09:49


import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, layers=1, bidirectional=True):
        super(Encoder, self).__init__()

        if bidirectional:
            self.directions = 2
        else:
            self.directions = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = layers
        # self.dropout = dropout
        self.embedder = nn.Embedding(input_size, hidden_size)
        #         self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=layers,
                            bidirectional=bidirectional, batch_first=False)
        self.fc = nn.Linear(hidden_size * self.directions, hidden_size)

    def forward(self, input_data, h_hidden, c_hidden):
        #         print(input_data)
        #         if torch.cuda.is_available():
        #             input_data.cuda()
        #         print(f'Input batch:\n{input_data.shape}\n*************')

        embedded_data = self.embedder(input_data)

        # embedded_data.to(gpu, torch.float)
        #         print(f'Embedded shape:\n{embedded_data.shape}\n*************')
        #         print(f'Embedded data:\n{embedded_data}\n*************')
        #         embedded_data = self.dropout(embedded_data)
        hiddens, outputs = self.lstm(embedded_data, (h_hidden, c_hidden))
        return hiddens, outputs

    """creates initial hidden states for encoder corresponding to batch size"""
    def create_init_hiddens(self, batch_size):
        h_hidden = Variable(torch.zeros(self.num_layers * self.directions,batch_size, self.hidden_size))
        c_hidden = Variable(torch.zeros(self.num_layers * self.directions,batch_size, self.hidden_size))
        # if torch.cuda.is_available():
        #     return h_hidden.cuda(), c_hidden.cuda()
        # else:
        return h_hidden, c_hidden





