import torch
import torch.nn as nn
import torch.nn.functional as funct


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, concat_output=True):

        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat_output = concat_output

        # Initialize weight matrix W
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)
        # Initialize attention nnet 'a'--> 2 layers and output is 1
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)

    def forward(self, h, adjacency):

        # Compute via multiplying W*h
        WH = torch.matmul(h, self.W)  # Size N (nodes) * out_features

        # For how good the coverage is use Travis's internal tool
        WH1 = torch.matmul(WH, self.a[:self.out_features, :])

        # Take second layer
        WH2 = torch.matmul(WH, self.a[self.out_features:, :])

        # Sum up everything to compute the alignment score
        e = nn.LeakyReLU(WH1 + WH2.T)

        # I'm avoiding an overflow here via substituting 0 values with -1e9
        attnt = torch.where(adjacency > 0, e, -1e9 * torch.ones_likes(e))

        attnt = funct.softmax(attnt, dim=1)
        hfirst = torch.matmul(attnt, WH)
        if self.concat_output:
            return funct.elu(hfirst)
        else:
            return hfirst
