import copy
import torch
import argparse
from datasets import SPMotif
from torch_geometric.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LEConv, BatchNorm, fps
from utils.mask import set_masks, clear_masks

import os
import numpy as np
import os.path as osp
from torch.autograd import grad
from datetime import datetime
from utils.helper import set_seed, args_print
from utils.get_subgraph import drop_info_return_full, relabel
from gnn import SPMotifNet
from gnn import SupConLoss


class QNet(nn.Module): # The q(.) for calculating IVs

    def __init__(self, ):
        super(QNet, self).__init__()
        self.convq1 = LEConv(in_channels=4, out_channels=args.channels)
        self.convq2 = LEConv(in_channels=args.channels, out_channels=args.channels)
        self.mlp = nn.Sequential(
            nn.Linear(args.channels * 2, args.channels * 4),
            nn.ReLU(),
            nn.Linear(args.channels * 4, 1)
        )

    def forward(self, data):
        
        q = self.convq1(data.x, data.edge_index, data.edge_attr.view(-1))
        q = self.convq2(q, data.edge_index, data.edge_attr.view(-1))

        row, col = data.edge_index
        edge_rep = torch.cat([q[row], q[col]], dim=-1)
        edge_weight = self.mlp(edge_rep).view(-1) # The edge wights as IVs

        return edge_weight 

class ProcessNet(nn.Module):

    def __init__(self, drop):
        super(ProcessNet, self).__init__()
        self.conv1 = LEConv(in_channels=4, out_channels=args.channels)
        self.conv2 = LEConv(in_channels=args.channels, out_channels=args.channels)
        self.mlp = nn.Sequential(
            nn.Linear(args.channels * 2, args.channels * 4),
            nn.ReLU(),
            nn.Linear(args.channels * 4, 1)
        )
        self.d = drop

    def forward(self, data, edge_score):
        # batch_norm
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr.view(-1)))
        x = self.conv2(x, data.edge_index, data.edge_attr.view(-1))

        (robust_edge_index, robust_edge_attr, robust_edge_weight), \
        (full_edge_index, full_edge_attr, full_edge_weight) = drop_info_return_full(data, edge_score, self.d) # r

        robust_x, robust_edge_index, robust_batch, _ = relabel(x, robust_edge_index, data.batch)
        full_x, full_edge_index, full_batch, _ = relabel(x, full_edge_index, data.batch)

        return (robust_x, robust_edge_index, robust_edge_attr, robust_edge_weight, robust_batch), \
               (full_x, full_edge_index, full_edge_attr, full_edge_weight, full_batch), \
               edge_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training RCGRL')
    parser.add_argument('--cuda', default=0, type=int, help='cuda device')
    parser.add_argument('--datadir', default='/home/gaohang/Researches/CaCL/RCGRL/data/', type=str,
                        help='directory for datasets.') # dataset location
    parser.add_argument('--epoch', default=300, type=int, help='training iterations')
    parser.add_argument('--seed', nargs='?', default=5, help='random seed')
    parser.add_argument('--channels', default=32, type=int, help='width of network')
    parser.add_argument('--bias', default='0.7', type=str, help='select bias extend')
    parser.add_argument('--pretrain', default=10, type=int, help='pretrain epoch')
    parser.add_argument('--lambda_Lc', default=1, type=float, help='lambda of Lc')
    parser.add_argument('--gamma_Lr', default=0.1, type=float, help='gamma of Lr')
    parser.add_argument('--tau_Lr', default=1000, type=float, help='gamma of Lr')
    parser.add_argument('--drop', default=0.75, type=float, help='percentage of data droped')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--net_lr', default=1e-3, type=float, help='learning rate for the predictor')
    args = parser.parse_args()
    # dataset
    num_classes = 3
    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
    train_dataset = SPMotif(osp.join(args.datadir, f'SPMotif-{args.bias}/'), mode='train')
    val_dataset = SPMotif(osp.join(args.datadir, f'SPMotif-{args.bias}/'), mode='val')
    test_dataset = SPMotif(osp.join(args.datadir, f'SPMotif-{args.bias}/'), mode='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    n_train_data, n_val_data = len(train_dataset), len(val_dataset)
    n_test_data = float(len(test_dataset))

    best_val = 0
    best_val_test = 0

    set_seed(args.seed)
    f_after = SPMotifNet(args.channels).to(device) # divided f(.) into two parts, before confounder removal and after
    q_net = QNet().to(device)
    f_before = ProcessNet(args.drop).to(device)

    model_optimizer = torch.optim.Adam(
        list(f_after.parameters()) +
        list(q_net.parameters()) +
        list(f_before.parameters()),
        lr=args.net_lr)
    CELoss = nn.CrossEntropyLoss(reduction="mean")
    EleCELoss = nn.CrossEntropyLoss(reduction="none")
    MSELoss = nn.MSELoss(reduction="none")
    SupConLoss_Func = SupConLoss()
    Cosine_Sim = nn.CosineSimilarity()


    def train_mode():
        f_after.train()
        f_before.train()
        q_net.eval()

    def train_q_mode():
        f_after.eval()
        f_before.eval()
        q_net.train()


    def val_mode():
        f_after.eval()
        f_before.eval()
        q_net.eval()

    def test_acc(loader, q_net, predictor):
        acc = 0
        for graph in loader:
            graph.to(device)

            edge_weight = q_net(graph)

            (robust_x, robust_edge_index, robust_edge_attr, robust_edge_weight, robust_batch), \
            (_, _, _, _, _), edge_score = f_before(graph,edge_weight)

            set_masks(robust_edge_weight, f_after)
            out = predictor(x=robust_x, edge_index=robust_edge_index,
                            edge_attr=robust_edge_attr, batch=robust_batch)
            clear_masks(f_after)

            acc += torch.sum(out.argmax(-1).view(-1) == graph.y.view(-1))
        acc = float(acc) / len(loader.dataset)
        return acc

    print(f"# Train: {n_train_data}  #Test: {n_test_data} #Val: {n_val_data}")
    cnt, last_val_acc = 0, 0

    for epoch in range(args.epoch):

        # udpate f
        all_loss, n_bw = 0, 0
        all_batched_loss, all_var_loss = 0, 0
        train_mode()
        for graph in train_loader:
            n_bw += 1
            graph.to(device)
            N = graph.num_graphs
            edge_score = q_net(graph)

            (robust_x, robust_edge_index, robust_edge_attr, robust_edge_weight, robust_batch), \
            (full_x, full_edge_index, full_edge_attr, full_edge_weight, full_batch), edge_score = f_before(graph, edge_score)

            set_masks(robust_edge_weight, f_after)
            robust_rep = f_after.get_graph_rep(
                x=robust_x, edge_index=robust_edge_index,
                edge_attr=robust_edge_attr, batch=robust_batch)
            robust_out = f_after.get_robust_pred(robust_rep)
            clear_masks(f_after)

            full_rep = f_after.get_graph_rep(
                x=full_x, edge_index=full_edge_index,
                edge_attr=full_edge_attr, batch=full_batch)

            robust_contrast_feature = robust_rep.unsqueeze(1).clone()
            full_contrast_feature = full_rep.unsqueeze(1).clone()
            contrast_feature = torch.cat((robust_contrast_feature, full_contrast_feature), 1)

            contrast_loss = (-1) * Cosine_Sim(robust_rep.detach(), full_rep)

            robust_loss = CELoss(robust_out, graph.y)
            robust_regular_mean = EleCELoss(robust_out, graph.y)

            robust_regular_target = robust_loss.expand_as(robust_regular_mean)
            robust_regular_mse = MSELoss(robust_regular_mean, robust_regular_target)

            robust_regular_mse_grouped_target = robust_regular_mean
            for i in range(num_classes):
                index = torch.nonzero(graph.y == float(i)).squeeze()
                robust_regular_mse_grouped_target[index] = torch.mean(robust_regular_mean[index])
            robust_regular = EleCELoss(robust_out, graph.y)

            robust_sp_feature = robust_rep.unsqueeze(1)
            robust_sp_feature = torch.cat((robust_sp_feature,robust_sp_feature),1)

            robust_sp_loss = SupConLoss_Func(robust_sp_feature, graph.y)

            robust_regular_mse_grouped = MSELoss(robust_regular, robust_regular_mse_grouped_target.detach())
            robust_regular_std = robust_regular.std()
            loss_r = robust_loss + args.lambda_Lc * contrast_loss.sum() # Lr

            all_batched_loss += loss_r

        all_batched_loss /= n_bw
        all_loss = all_batched_loss
        model_optimizer.zero_grad()
        all_loss.backward()
        model_optimizer.step()

        # update q
        all_loss, n_bw = 0, 0
        all_batched_loss, all_var_loss = 0, 0
        train_q_mode()
        for graph in train_loader:
            n_bw += 1
            graph.to(device)
            N = graph.num_graphs
            edge_score = q_net(graph)

            (robust_x, robust_edge_index, robust_edge_attr, robust_edge_weight, robust_batch), \
            (full_x, full_edge_index, full_edge_attr, full_edge_weight, full_batch), edge_score = f_before(graph,
                                                                                                          edge_score)
            set_masks(robust_edge_weight, f_after)
            robust_rep = f_after.get_graph_rep(
                x=robust_x, edge_index=robust_edge_index,
                edge_attr=robust_edge_attr, batch=robust_batch)
            robust_out = f_after.get_robust_pred(robust_rep)
            clear_masks(f_after)

            robust_loss = CELoss(robust_out, graph.y)
            robust_regular_mean = EleCELoss(robust_out, graph.y)

            robust_regular_target = robust_loss.expand_as(robust_regular_mean)
            robust_regular_mse = MSELoss(robust_regular_mean, robust_regular_target)

            robust_regular_mse_grouped_target = robust_regular_mean
            for i in range(num_classes):
                index = torch.nonzero(graph.y == float(i)).squeeze()
                robust_regular_mse_grouped_target[index] = torch.mean(robust_regular_mean[index])
            robust_regular = EleCELoss(robust_out, graph.y)

            robust_sp_feature = robust_rep.unsqueeze(1)
            robust_sp_feature = torch.cat((robust_sp_feature,robust_sp_feature),1)
            robust_sp_loss = SupConLoss_Func(robust_sp_feature, graph.y)

            robust_weight_loss_target = robust_rep.clone()
            for i in range(num_classes):
                index = torch.nonzero(graph.y == float(i)).squeeze()
                robust_weight_loss_target[index] = torch.mean(robust_rep[index].clone(),dim = 0)
            robust_weight = Cosine_Sim(robust_rep, robust_weight_loss_target)

            robust_sim_all_target = robust_rep.clone()
            robust_sim_all_target[:] = torch.mean(robust_rep.clone(),dim = 0)
            robust_sim_all = Cosine_Sim(robust_rep, robust_sim_all_target)
            robust_sim_all = (1-robust_sim_all)/2
            robust_sim_all = robust_sim_all/robust_sim_all.shape[0]
            robust_sim_all = robust_sim_all.sum()

            if robust_sim_all < 0.01:
                sim_all_e = -1 * args.tau_Lr
            else:
                sim_all_e = 0

            correct_index = torch.nonzero(robust_out.argmax(-1).view(-1) == graph.y.view(-1)).squeeze()
            wrong_index = torch.nonzero(robust_out.argmax(-1).view(-1) != graph.y.view(-1)).squeeze()
            robust_weight[correct_index] = 1 - robust_weight[correct_index]
            robust_weight[wrong_index] = robust_weight[wrong_index] + 1
            robust_weight = robust_weight/2
            robust_weight = robust_weight.pow(args.gamma_Lr).detach()
            robust_regular_weighted = robust_regular * robust_weight

            loss_c = robust_regular_weighted.sum() + sim_all_e * robust_sim_all # Lc

            all_batched_loss += loss_c

        all_batched_loss /= n_bw
        all_loss = all_batched_loss
        model_optimizer.zero_grad()
        all_loss.backward()
        model_optimizer.step()




        val_mode()
        with torch.no_grad():

            train_acc = test_acc(train_loader, q_net, f_after)
            val_acc = test_acc(val_loader, q_net, f_after)
            robust_acc = 0.
            for graph in test_loader:
                graph.to(device)
                edge_score = q_net(graph)

                (robust_x, robust_edge_index, robust_edge_attr, robust_edge_weight, robust_batch), \
                (full_x, full_edge_index, full_edge_attr, full_edge_weight, full_batch), edge_score = f_before(graph,
                                                                                                              edge_score)

                set_masks(robust_edge_weight, f_after)
                robust_out = f_after(
                    x=robust_x, edge_index=robust_edge_index,
                    edge_attr=robust_edge_attr, batch=robust_batch)
                clear_masks(f_after)
                robust_acc += torch.sum(robust_out.argmax(-1).view(-1) == graph.y.view(-1)) / n_test_data

            print("Epoch {:3d}  all_loss:{:2.3f}  "
                        "Train_ACC:{:.3f} Test_ACC{:.3f}  Val_ACC:{:.3f}  ".format(
                epoch, all_loss,
                train_acc, robust_acc, val_acc
                ))

        print("val_acc:", val_acc, "last_val_acc:", last_val_acc, "cnt:", cnt)

        if epoch >= args.pretrain:
            if val_acc < last_val_acc:
                cnt += 1
            else:
                cnt = 0
                last_val_acc = val_acc
        if cnt >= 5:
            print("Early Stop!")
            break
        if val_acc > best_val:
            # print("val_acc > best_val", val_acc, ">", best_val)
            best_val = val_acc
            best_val_test = robust_acc

    print("best val is", best_val)
    print("best test is", best_val_test)
