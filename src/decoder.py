# -*- coding: utf-8 -*-
# @Author : Prabhu Appalapuri<prabhu.appalapuri@gmail.com>
# @Time : 20.02.20 09:54


import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, layers=1, dropout=0.1, bidirectional=True):
        super(Decoder, self).__init__()
        if bidirectional:
            self.directions = 2
        else:
            self.directions = 1
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = layers
        self.dropout = dropout
        self.embedder = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.score_learner = nn.Linear(hidden_size * self.directions,
                                       hidden_size * self.directions)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=layers, dropout=dropout,
                            bidirectional=bidirectional, batch_first=False)
        self.context_combiner = nn.Linear((hidden_size * self.directions)
                                          + (hidden_size * self.directions), hidden_size)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(hidden_size, output_size)
        self.soft = nn.Softmax(dim=1)
        self.log_soft = nn.LogSoftmax(dim=1)

    def forward(self, input_data, h_hidden, c_hidden, encoder_hiddens):

        embedded_data = self.embedder(input_data)
        embedded_data = self.dropout(embedded_data)

        batch_size = embedded_data.shape[1]
        hiddens, outputs = self.lstm(embedded_data, (h_hidden, c_hidden))
        top_hidden = outputs[0].view(self.num_layers, self.directions, hiddens.shape[1], self.hidden_size)[self.num_layers - 1]
        top_hidden = top_hidden.permute(1, 2, 0).contiguous().view(batch_size, -1, 1)

        prep_scores = self.score_learner(encoder_hiddens.permute(1, 0, 2))
        scores = torch.bmm(prep_scores, top_hidden)
        attn_scores = self.soft(scores)
        con_mat = torch.bmm(encoder_hiddens.permute(1, 2, 0), attn_scores)
        h_tilde = self.tanh(self.context_combiner(torch.cat((con_mat, top_hidden), dim=1).view(batch_size, -1)))
        pred = self.output(h_tilde)
        pred = self.log_soft(pred)
        return pred, outputs



