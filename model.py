import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import SAGEConv

from config import Config

config = Config()


class GCN(nn.Module):
    def __init__(self, embedding_size, h_feats, dropout):
        super(GCN, self).__init__()
        self.gcn_out_dim = 4 * h_feats
        self.embedding = nn.Embedding(256 + 1, embedding_size)
        self.gcn1 = SAGEConv(embedding_size, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))
        self.gcn2 = SAGEConv(h_feats, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))
        self.gcn3 = SAGEConv(h_feats, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))
        self.gcn4 = SAGEConv(h_feats, h_feats, 'mean', activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))

    def forward(self, g, in_feat):
        in_feat = in_feat.long()
        h = self.embedding(in_feat.view(-1))
        h1 = self.gcn1(g, h)
        h2 = self.gcn2(g, h1)
        h3 = self.gcn3(g, h2)
        h4 = self.gcn4(g, h3)
        g.ndata['h'] = torch.cat((h1, h2, h3, h4), dim=1)
        g_vec = dgl.mean_nodes(g, 'h')

        return g_vec


class Cross_Gated_Info_Filter(nn.Module):
    def __init__(self, in_size):
        super(Cross_Gated_Info_Filter, self).__init__()
        self.filter1 = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.PReLU(config.FLOW_PAD_TRUNC_LENGTH),
            nn.Linear(in_size, in_size)
        )
        self.filter2 = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.PReLU(config.FLOW_PAD_TRUNC_LENGTH),
            nn.Linear(in_size, in_size)
        )

    def forward(self, x, y):
        ori_x = x
        ori_y = y
        z1 = self.filter1(x).sigmoid() * ori_y
        z2 = self.filter2(y).sigmoid() * ori_x

        return torch.cat([z1, z2], dim=-1)


class MixTemporalGNN(nn.Module):
    def __init__(self, num_classes, embedding_size=64, h_feats=128, dropout=0.2, downstream_dropout=0.0):
        super(MixTemporalGNN, self).__init__()
        self.header_graphConv = GCN(embedding_size=embedding_size, h_feats=h_feats, dropout=dropout)
        self.payload_graphConv = GCN(embedding_size=embedding_size, h_feats=h_feats, dropout=dropout)
        self.gcn_out_dim = 4 * h_feats
        self.gated_filter = Cross_Gated_Info_Filter(in_size=self.gcn_out_dim)
        self.rnn = nn.LSTM(input_size=self.gcn_out_dim * 2, hidden_size=self.gcn_out_dim * 2, num_layers=2, bidirectional=True, dropout=downstream_dropout)
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.gcn_out_dim * 4, out_features=self.gcn_out_dim),
            nn.PReLU(self.gcn_out_dim)
        )
        self.cls = nn.Linear(in_features=self.gcn_out_dim, out_features=num_classes)

    def forward(self, header_graph_data, payload_graph_data, labels):
        header_gcn_out = self.header_graphConv(header_graph_data, header_graph_data.ndata['feat']).reshape(
            labels.shape[0], config.FLOW_PAD_TRUNC_LENGTH, -1)
        payload_gcn_out = self.payload_graphConv(payload_graph_data, payload_graph_data.ndata['feat']).reshape(
            labels.shape[0], config.FLOW_PAD_TRUNC_LENGTH, -1)
        gcn_out = self.gated_filter(header_gcn_out, payload_gcn_out)
        gcn_out = gcn_out.transpose(0, 1)
        _, (h_n, _) = self.rnn(gcn_out)
        rnn_out = torch.cat((h_n[-1], h_n[-2]), dim=1)
        out = self.fc(rnn_out)
        out = self.cls(out)

        return out


if __name__ == '__main__':
    pass
