import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import A3TGCN2 # this version accepts batch inputs
from torch_geometric_temporal.nn.recurrent import TGCN2 # so does this

    

class TrafficA3TGCNSingleShot(nn.Module):
    """
    accepts batches
    """
    def __init__(self, node_features, periods_in, periods_out, dropout_rate, batch_size):
        super(TrafficA3TGCNSingleShot, self).__init__()
        # encoder
        self.tgnn = A3TGCN2(in_channels=node_features, 
                           out_channels=32, 
                           periods=periods_in,
                           batch_size = batch_size)
        
        # this is for single shot prediction. 
        #Project each of the node vectors of length 32 to a vector of period_out numbers 
        #that are supposed to be the predictions for the next time steps
        self.linear1 = nn.Linear(32, 16)
        self.linear2 = nn.Linear(16, periods_out)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_features):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index) # returns the final hidden state for each node
        # print(h.shape)
        h = F.tanh(h)
        h = self.linear1(h)
        h = F.tanh(h)
        h = self.dropout(h)
        h = self.linear2(h)
        # print(h.shape)
        return h


class TrafficA3TGCN_LSTM(nn.Module):
    def __init__(self, node_features, node_out_features, periods_in, periods_out, dropout_rate, batch_size):
        super(TrafficA3TGCN_LSTM, self).__init__()
        self.periods_out = periods_out
        # encoder
        self.tgnn = A3TGCN2(in_channels=node_features, 
                           out_channels=16, 
                           periods=periods_in,
                          batch_size = batch_size)
        # decoder
        self.lstm = nn.LSTM(input_size = 16, hidden_size = 16, num_layers = 2)
        self.linear = nn.Linear(16, node_out_features)

    def forward(self, x, edge_index, edge_features):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index, edge_features) # returns the final hidden state for each node

        h.unsqueeze_(-1)

        h = h.expand(h.shape[0], h.shape[1],h.shape[2],self.periods_out)

        h = torch.permute(h, (0, 3,1,2))

        outputs = []
        for i in range(h.shape[0]):
            input_data = h[i,:,:,:]
            out, _ = self.lstm(input_data)
            outputs.append(self.linear(out))
        out = torch.stack(outputs)

        out = torch.permute(out, (0,2,3,1))

        return out
