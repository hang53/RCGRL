import argparse
import os.path as osp
import random
import time
import sys
import torch
from torch.nn import Linear, ReLU, ModuleList, Softmax, CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GCNConv, BatchNorm, MessagePassing, global_mean_pool ,global_max_pool, LEConv
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from .overloader import overload


class SPMotifNet(torch.nn.Module):
    def __init__(self, in_channels, hid_channels=64, num_classes=3, num_unit=2):
        super().__init__()

        self.num_unit = num_unit
        self.node_emb = Linear(in_channels, hid_channels)

        self.convs = ModuleList()
        self.relus = ModuleList()
        for i in range(num_unit):
            conv = LEConv(in_channels=hid_channels, out_channels=hid_channels)
            self.convs.append(conv)
            self.relus.append(ReLU())
        
        self.robust_mlp = torch.nn.Sequential(
            Linear(hid_channels, 2*hid_channels),
            ReLU(),
            Linear(2*hid_channels, num_classes)
        )

        self.conf_mlp = torch.nn.Sequential(
            Linear(hid_channels, 2*hid_channels),
            ReLU(),
            Linear(2*hid_channels, 3)
        )
        self.cq = Linear(3, 3)
        self.conf_fw = torch.nn.Sequential(
            self.conf_mlp,
            self.cq
        )
    
    @overload
    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return self.get_robust_pred(graph_x)
    
    @overload
    def get_node_reps(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        for conv, ReLU in \
                zip(self.convs, self.relus):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = ReLU(x)
        node_x = x
        return node_x
    
    @overload
    def get_graph_rep(self, x, edge_index, edge_attr, batch):

        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_robust_pred(self, robust_graph_x):
        pred = self.robust_mlp(robust_graph_x)
        return pred

    def get_conf_pred(self, conf_graph_x):
        pred = self.conf_fw(conf_graph_x)
        return pred

    def get_comb_pred(self, robust_graph_x, conf_graph_x):
        robust_pred = self.robust_mlp(robust_graph_x)
        conf_pred = self.conf_mlp(conf_graph_x).detach()
        return torch.sigmoid(conf_pred) * robust_pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)


class SPMotifNet_LEGNN(torch.nn.Module):
    def __init__(self, in_channels, hid_channels=64, num_classes=3, num_unit=2):
        super().__init__()

        self.num_unit = num_unit
        self.node_emb = Linear(in_channels, hid_channels)

        self.convs = ModuleList()
        self.relus = ModuleList()
        for i in range(num_unit):
            conv = LEConv(in_channels=hid_channels, out_channels=hid_channels)
            self.convs.append(conv)
            self.relus.append(ReLU())

        self.robust_mlp = torch.nn.Sequential(
            Linear(hid_channels, 2 * hid_channels),
            ReLU(),
            Linear(2 * hid_channels, num_classes)
        )

        self.conf_mlp = torch.nn.Sequential(
            Linear(hid_channels, 2 * hid_channels),
            ReLU(),
            Linear(2 * hid_channels, 3)
        )
        self.cq = Linear(3, 3)
        self.conf_fw = torch.nn.Sequential(
            self.conf_mlp,
            self.cq
        )

    @overload
    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return node_x, graph_x, self.get_robust_pred(graph_x)

    @overload
    def get_node_reps(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        for conv, ReLU in \
                zip(self.convs, self.relus):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = ReLU(x)
        node_x = x
        return node_x

    @overload
    def get_graph_rep(self, x, edge_index, edge_attr, batch):

        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_graph_rep_with_node_features(self, node_x, batch):
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_robust_pred(self, robust_graph_x):
        pred = self.robust_mlp(robust_graph_x)
        return pred

    def get_conf_pred(self, conf_graph_x):
        pred = self.conf_fw(conf_graph_x)
        return pred

    def get_comb_pred(self, robust_graph_x, conf_graph_x):
        robust_pred = self.robust_mlp(robust_graph_x)
        conf_pred = self.conf_mlp(conf_graph_x).detach()
        return torch.sigmoid(conf_pred) * robust_pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)


class SPMotifNet_ori_x(torch.nn.Module):
    def __init__(self, in_channels, hid_channels=64, num_classes=3, num_unit=2):
        super().__init__()

        self.num_unit = num_unit
        self.node_emb = Linear(4, 32)

        self.convs = ModuleList()
        self.relus = ModuleList()
        conv = LEConv(in_channels=32, out_channels=hid_channels)
        self.convs.append(conv)
        self.relus.append(ReLU())
        for i in range(num_unit):
            conv = LEConv(in_channels=hid_channels, out_channels=hid_channels)
            self.convs.append(conv)
            self.relus.append(ReLU())

        self.robust_mlp = torch.nn.Sequential(
            Linear(hid_channels, 2 * hid_channels),
            ReLU(),
            Linear(2 * hid_channels, num_classes)
        )

        self.conf_mlp = torch.nn.Sequential(
            Linear(hid_channels, 2 * hid_channels),
            ReLU(),
            Linear(2 * hid_channels, 3)
        )
        self.cq = Linear(3, 3)
        self.conf_fw = torch.nn.Sequential(
            self.conf_mlp,
            self.cq
        )

    @overload
    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return self.get_robust_pred(graph_x)

    @overload
    def get_node_reps(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        for conv, ReLU in \
                zip(self.convs, self.relus):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = ReLU(x)
        node_x = x
        return node_x

    @overload
    def get_graph_rep(self, x, edge_index, edge_attr, batch):

        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_robust_pred(self, robust_graph_x):
        pred = self.robust_mlp(robust_graph_x)
        return pred

    def get_conf_pred(self, conf_graph_x):
        pred = self.conf_fw(conf_graph_x)
        return pred

    def get_comb_pred(self, robust_graph_x, conf_graph_x):
        robust_pred = self.robust_mlp(robust_graph_x)
        conf_pred = self.conf_mlp(conf_graph_x).detach()
        return torch.sigmoid(conf_pred) * robust_pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)