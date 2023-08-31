import torch
import math
import math
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import (remove_self_loops, degree, 
                                   batched_negative_sampling)
from torch_geometric.utils.num_nodes import maybe_num_nodes

MAX_DIAM=100


def get_neg_edge_index(g):
    neg_edge_index = batched_negative_sampling(edge_index=g.edge_index,
                                               batch=g.batch,
                                               num_neg_samples=None,
                                               force_undirected=False)
    neg_edge_index, _ = remove_self_loops(neg_edge_index)
    return neg_edge_index


def split_batch(g):
    split = degree(g.batch[g.edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(g.edge_index, split, dim=1)
    num_nodes = degree(g.batch, dtype=torch.long)
    cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
    num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(g.x.device)
    cum_edges = torch.cat([g.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])

    return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges

        
def drop_info(data, edge_score, ratio):
    robust_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    robust_edge_weight = torch.tensor([]).to(data.x.device)
    robust_edge_attr = torch.tensor([]).to(data.x.device)
    full_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    full_edge_weight = torch.tensor([]).to(data.x.device)
    full_edge_attr = torch.tensor([]).to(data.x.device)

    edge_indices, _, _, num_edges, cum_edges = split_batch(data)
    for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
        n_reserve =  int(ratio * N)
        edge_attr = data.edge_attr[C:C+N]
        single_mask = edge_score[C:C+N]
        single_mask_detach = edge_score[C:C+N].detach().cpu().numpy()
        rank = np.argpartition(-single_mask_detach, n_reserve)
        idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]

        robust_edge_index = torch.cat([robust_edge_index, edge_index[:, idx_reserve]], dim=1)
        full_edge_index = torch.cat([full_edge_index, edge_index[:, :]], dim=1)

        robust_edge_weight = torch.cat([robust_edge_weight, single_mask[idx_reserve]])
        full_edge_weight = torch.cat([full_edge_weight,  single_mask[:]])

        robust_edge_attr = torch.cat([robust_edge_attr, edge_attr[idx_reserve]])
        full_edge_attr = torch.cat([full_edge_attr, edge_attr[:]])
    return (robust_edge_index, robust_edge_attr, robust_edge_weight), \
        (full_edge_index, full_edge_attr, full_edge_weight)



def drop_info_return_full(data, edge_score, d, require_edge_reserve_index = False):
    robust_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    robust_edge_weight = torch.tensor([]).to(data.x.device)
    edge_reserve_index = torch.LongTensor([]).to(data.x.device)
    robust_edge_attr = torch.tensor([]).to(data.x.device)
    
    full_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    full_edge_weight = torch.tensor([]).to(data.x.device)
    full_edge_attr = torch.tensor([]).to(data.x.device)

    edge_indices, _, _, num_edges, cum_edges = split_batch(data)
    # counter = 0
    for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
        n_reserve = int((1-d) * N)
        edge_attr = data.edge_attr[C:C+N]
        single_mask = edge_score[C:C+N]
        # single_mask = F.sigmoid(edge_score[C:C + N] * 100)

        # single_mask = single_mask.pow(1)
        single_mask_detach = edge_score[C:C+N].detach().cpu().numpy()
        rank = np.argpartition(-single_mask_detach, n_reserve)
        idx_reserve = rank[:n_reserve]
        # idx_reserve = rank

        robust_edge_index = torch.cat([robust_edge_index, edge_index[:, idx_reserve]], dim=1)
        # robust_edge_index = torch.cat([robust_edge_index, edge_index[:, :]], dim=1)
        full_edge_index = torch.cat([full_edge_index, edge_index[:, :]], dim=1)

        # robust_edge_weight = torch.cat([robust_edge_weight, single_mask[idx_reserve]])
        robust_edge_weight = torch.cat([robust_edge_weight, single_mask[idx_reserve]])
        idx_reserve_tn = torch.from_numpy(idx_reserve).cuda()
        # print("idx_reserve is ", idx_reserve)
        # print("edge_reserve_index is ", edge_reserve_index)
        # counter = counter + 1
        # print("counter is", counter)
        # if counter == 64:
        #     print(" ")
        edge_reserve_index = torch.cat([edge_reserve_index, idx_reserve_tn + C])
        # print("edge_reserve_index is ", edge_reserve_index)
        full_edge_weight = torch.cat([full_edge_weight,  single_mask])



        # robust_edge_attr = torch.cat([robust_edge_attr, edge_attr[idx_reserve]])
        robust_edge_attr = torch.cat([robust_edge_attr, edge_attr[idx_reserve]])
        full_edge_attr = torch.cat([full_edge_attr, edge_attr])

        # print("edge_reserve_index is ", edge_reserve_index)
        # print("C is ", C)

    if require_edge_reserve_index:
        return (robust_edge_index, robust_edge_attr, robust_edge_weight, edge_reserve_index), \
               (full_edge_index, full_edge_attr, full_edge_weight)
    else:
        return (robust_edge_index, robust_edge_attr, robust_edge_weight), \
            (full_edge_index, full_edge_attr, full_edge_weight)

def drop_info_return_full_min1(data, edge_score, d):
    robust_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    robust_edge_weight = torch.tensor([]).to(data.x.device)
    robust_edge_attr = torch.tensor([]).to(data.x.device)
    full_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    full_edge_weight = torch.tensor([]).to(data.x.device)
    full_edge_attr = torch.tensor([]).to(data.x.device)

    edge_indices, _, _, num_edges, cum_edges = split_batch(data)
    for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
        n_reserve = max(1,int((1-d) * N))
        edge_attr = data.edge_attr[C:C+N]
        single_mask = edge_score[C:C+N]
        single_mask_detach = edge_score[C:C+N].detach().cpu().numpy()
        rank = np.argpartition(-single_mask_detach, n_reserve)
        idx_reserve = rank[:n_reserve]

        robust_edge_index = torch.cat([robust_edge_index, edge_index[:, idx_reserve]], dim=1)
        full_edge_index = torch.cat([full_edge_index, edge_index[:, :]], dim=1)

        robust_edge_weight = torch.cat([robust_edge_weight, single_mask[idx_reserve]])
        full_edge_weight = torch.cat([full_edge_weight,  single_mask])

        robust_edge_attr = torch.cat([robust_edge_attr, edge_attr[idx_reserve]])
        full_edge_attr = torch.cat([full_edge_attr, edge_attr])




    return (robust_edge_index, robust_edge_attr, robust_edge_weight), \
        (full_edge_index, full_edge_attr, full_edge_weight)

def drop_info_return_full_min1_with_edge_remove(data, edge_score, d, enable_edge_remove, index):
    robust_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    robust_edge_weight = torch.tensor([]).to(data.x.device)
    robust_edge_attr = torch.tensor([]).to(data.x.device)
    full_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    full_edge_weight = torch.tensor([]).to(data.x.device)
    full_edge_attr = torch.tensor([]).to(data.x.device)

    edge_indices, _, _, num_edges, cum_edges = split_batch(data)

    if enable_edge_remove == False:
        for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):

            n_reserve = max(1,int((1-d) * N))
            edge_attr = data.edge_attr[C:C+N]
            single_mask = edge_score[C:C+N]
            single_mask_detach = edge_score[C:C+N].detach().cpu().numpy()
            rank = np.argpartition(-single_mask_detach, n_reserve)
            idx_reserve = rank[:n_reserve]

            robust_edge_index = torch.cat([robust_edge_index, edge_index[:, idx_reserve]], dim=1)
            full_edge_index = torch.cat([full_edge_index, edge_index[:, :]], dim=1)

            robust_edge_weight = torch.cat([robust_edge_weight, single_mask[idx_reserve]])
            full_edge_weight = torch.cat([full_edge_weight,  single_mask])

            robust_edge_attr = torch.cat([robust_edge_attr, edge_attr[idx_reserve]])
            full_edge_attr = torch.cat([full_edge_attr, edge_attr])

        return (robust_edge_index, robust_edge_attr, robust_edge_weight), \
            (full_edge_index, full_edge_attr, full_edge_weight)

    else:
        counter = 0
        total_edges = 0
        for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):

            if index == 0:
                print("graph ", counter, " N is:", N, "C is ", C)

                print(edge_index - edge_index[0][0])

                total_edges = total_edges + N
            counter = counter + 1

            if N > index:
                idx_keep = np.linspace(0, N.clone().detach().cpu() - 1, N.clone().detach().cpu())
                # idx_reserve = np.delete(idx_reserve, index)

                single_mask = edge_score[C:C + N]

                start_node = edge_index[0, index]
                stop_node = edge_index[1, index]

                adj_index_start1 = (edge_index[0] == start_node).nonzero(as_tuple=False)
                adj_index_start1 = adj_index_start1.detach().cpu().numpy()

                adj_index_stop1 = (edge_index[0] == stop_node).nonzero(as_tuple=False)
                adj_index_stop1 = adj_index_stop1.detach().cpu().numpy()

                adj_index_start2 = (edge_index[1] == start_node).nonzero(as_tuple=False)
                adj_index_start2 = adj_index_start2.detach().cpu().numpy()

                adj_index_stop2 = (edge_index[1] == stop_node).nonzero(as_tuple=False)
                adj_index_stop2 = adj_index_stop2.detach().cpu().numpy()

                adj_index = np.concatenate([adj_index_start1, adj_index_stop1, adj_index_start2, adj_index_stop2],
                                           axis=0)

                idx_keep = np.delete(idx_keep, adj_index)

                idx_keep = idx_keep.astype(int)

                # # index_tensor = torch.from_numpy(idx_keep)
                # edge_attr_edge_removed = data.edge_attr[idx_keep]
                #
                # edge_index_edge_removed = edge_index[:, idx_keep]
                #
                # edge_weight_edge_removed = single_mask[idx_keep]
                #
                # N_ = idx_keep.size
                #
                # n_keep = max(1, int(ratio * N))
                #
                # # print("n_reserve is", n_reserve)

                n_reserve = max(1, int((1-d) * N))
                edge_attr = data.edge_attr[C:C + N]
                single_mask = edge_score[C:C + N]
                single_mask_detach = edge_score[C:C + N].detach().cpu().numpy()
                rank = np.argpartition(-single_mask_detach, n_reserve)
                idx_reserve = rank[:n_reserve]

                idx_reserve = np.intersect1d(idx_reserve, idx_keep)

                robust_edge_index = torch.cat([robust_edge_index, edge_index[:, idx_reserve]], dim=1)
                full_edge_index = torch.cat([full_edge_index, edge_index[:, idx_keep]], dim=1)

                robust_edge_weight = torch.cat([robust_edge_weight, single_mask[idx_reserve]])
                full_edge_weight = torch.cat([full_edge_weight, single_mask[idx_keep]])

                robust_edge_attr = torch.cat([robust_edge_attr, edge_attr[idx_reserve]])
                full_edge_attr = torch.cat([full_edge_attr, edge_attr[idx_keep]])

                # edge_index = edge_index_edge_removed
                # edge_attr = edge_attr_edge_removed
                # single_mask = edge_weight_edge_removed
                # single_mask_detach = edge_weight_edge_removed.detach().cpu().numpy()
                # rank = np.argpartition(-single_mask_detach, n_reserve)
                # idx_reserve = rank[:n_reserve]
                #
                # robust_edge_index = torch.cat([robust_edge_index, edge_index_edge_removed[:, idx_reserve]], dim=1)
                # full_edge_index = torch.cat([full_edge_index, edge_index_edge_removed[:, :]], dim=1)
                #
                # robust_edge_weight = torch.cat([robust_edge_weight, single_mask[idx_reserve]])
                # full_edge_weight = torch.cat([full_edge_weight, single_mask])
                #
                # robust_edge_attr = torch.cat([robust_edge_attr, edge_attr[idx_reserve]])
                # full_edge_attr = torch.cat([full_edge_attr, edge_attr])
            else:
                n_reserve = max(1, int((1-d) * N))
                edge_attr = data.edge_attr[C:C + N]
                single_mask = edge_score[C:C + N]
                single_mask_detach = edge_score[C:C + N].detach().cpu().numpy()
                rank = np.argpartition(-single_mask_detach, n_reserve)
                idx_reserve = rank[:n_reserve]

                robust_edge_index = torch.cat([robust_edge_index, edge_index[:, idx_reserve]], dim=1)
                full_edge_index = torch.cat([full_edge_index, edge_index[:, :]], dim=1)

                robust_edge_weight = torch.cat([robust_edge_weight, single_mask[idx_reserve]])
                full_edge_weight = torch.cat([full_edge_weight, single_mask])

                robust_edge_attr = torch.cat([robust_edge_attr, edge_attr[idx_reserve]])
                full_edge_attr = torch.cat([full_edge_attr, edge_attr])

        return (robust_edge_index, robust_edge_attr, robust_edge_weight), \
               (full_edge_index, full_edge_attr, full_edge_weight)

def drop_info_return_full_min1_motivate(data, edge_score, d, enable_edge_remove, index):
    robust_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    robust_edge_weight = torch.tensor([]).to(data.x.device)
    robust_edge_attr = torch.tensor([]).to(data.x.device)
    full_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    full_edge_weight = torch.tensor([]).to(data.x.device)
    full_edge_attr = torch.tensor([]).to(data.x.device)

    edge_indices, _, _, num_edges, cum_edges = split_batch(data)

    if enable_edge_remove == False:
        for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):

            n_reserve = max(1,int((1-d) * N))
            edge_attr = data.edge_attr[C:C+N]
            single_mask = edge_score[C:C+N]
            single_mask_detach = edge_score[C:C+N].detach().cpu().numpy()
            # rank = np.argpartition(-single_mask_detach, n_reserve)
            idx_reserve = np.arange(0,n_reserve,1)

            robust_edge_index = torch.cat([robust_edge_index, edge_index[:, idx_reserve]], dim=1)
            full_edge_index = torch.cat([full_edge_index, edge_index[:, :]], dim=1)

            robust_edge_weight = torch.cat([robust_edge_weight, single_mask[idx_reserve]])
            full_edge_weight = torch.cat([full_edge_weight,  single_mask])

            robust_edge_attr = torch.cat([robust_edge_attr, edge_attr[idx_reserve]])
            full_edge_attr = torch.cat([full_edge_attr, edge_attr])

        return (robust_edge_index, robust_edge_attr, robust_edge_weight), \
            (full_edge_index, full_edge_attr, full_edge_weight)

    else:
        counter = 0
        total_edges = 0
        for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):

            if index == 0:
                print("graph ", counter, " N is:", N, "C is ", C)

                print(edge_index - edge_index[0][0])

                total_edges = total_edges + N
            counter = counter + 1

            if N > index:
                idx_keep = np.linspace(0, N.clone().detach().cpu() - 1, N.clone().detach().cpu())

                single_mask = edge_score[C:C + N]

                start_node = edge_index[0, index]
                stop_node = edge_index[1, index]

                adj_index_start1 = (edge_index[0] == start_node).nonzero(as_tuple=False)
                adj_index_start1 = adj_index_start1.detach().cpu().numpy()

                adj_index_stop1 = (edge_index[0] == stop_node).nonzero(as_tuple=False)
                adj_index_stop1 = adj_index_stop1.detach().cpu().numpy()

                adj_index_start2 = (edge_index[1] == start_node).nonzero(as_tuple=False)
                adj_index_start2 = adj_index_start2.detach().cpu().numpy()

                adj_index_stop2 = (edge_index[1] == stop_node).nonzero(as_tuple=False)
                adj_index_stop2 = adj_index_stop2.detach().cpu().numpy()

                adj_index = np.concatenate([adj_index_start1, adj_index_stop1, adj_index_start2, adj_index_stop2],
                                           axis=0)

                idx_keep = np.delete(idx_keep, adj_index)

                idx_keep = idx_keep.astype(int)

                n_reserve = max(1, int((1-d) * N))
                edge_attr = data.edge_attr[C:C + N]
                single_mask = edge_score[C:C + N]
                single_mask_detach = edge_score[C:C + N].detach().cpu().numpy()
                rank = np.argpartition(-single_mask_detach, n_reserve)
                idx_reserve = rank[:n_reserve]

                idx_reserve = np.intersect1d(idx_reserve, idx_keep)

                robust_edge_index = torch.cat([robust_edge_index, edge_index[:, idx_reserve]], dim=1)
                full_edge_index = torch.cat([full_edge_index, edge_index[:, idx_keep]], dim=1)

                robust_edge_weight = torch.cat([robust_edge_weight, single_mask[idx_reserve]])
                full_edge_weight = torch.cat([full_edge_weight, single_mask[idx_keep]])

                robust_edge_attr = torch.cat([robust_edge_attr, edge_attr[idx_reserve]])
                full_edge_attr = torch.cat([full_edge_attr, edge_attr[idx_keep]])
            else:
                n_reserve = max(1, int((1-d) * N))
                edge_attr = data.edge_attr[C:C + N]
                single_mask = edge_score[C:C + N]
                single_mask_detach = edge_score[C:C + N].detach().cpu().numpy()
                rank = np.argpartition(-single_mask_detach, n_reserve)
                idx_reserve = rank[:n_reserve]

                robust_edge_index = torch.cat([robust_edge_index, edge_index[:, idx_reserve]], dim=1)
                full_edge_index = torch.cat([full_edge_index, edge_index[:, :]], dim=1)

                robust_edge_weight = torch.cat([robust_edge_weight, single_mask[idx_reserve]])
                full_edge_weight = torch.cat([full_edge_weight, single_mask])

                robust_edge_attr = torch.cat([robust_edge_attr, edge_attr[idx_reserve]])
                full_edge_attr = torch.cat([full_edge_attr, edge_attr])

        return (robust_edge_index, robust_edge_attr, robust_edge_weight), \
               (full_edge_index, full_edge_attr, full_edge_weight)


def drop_info_return_edge_removal(data, index):
    robust_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    robust_edge_attr = torch.tensor([]).to(data.x.device)

    edge_indices, _, _, num_edges, cum_edges = split_batch(data)
    counter = 0
    total_edges = 0
    for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):

        if index == 0:
            print("graph ", counter, " N is:", N , "C is ", C )

            print(edge_index -edge_index[0][0])

            total_edges = total_edges + N
        counter = counter + 1


        if N > index:
            idx_reserve = np.linspace(0, N.clone().detach().cpu() - 1, N.clone().detach().cpu())
            # idx_reserve = np.delete(idx_reserve, index)

            start_node = edge_index[0,index]
            stop_node = edge_index[1,index]

            adj_index_start1 = (edge_index[0] == start_node).nonzero(as_tuple = False)
            adj_index_start1 = adj_index_start1.detach().cpu().numpy()

            adj_index_stop1 = (edge_index[0] == stop_node).nonzero(as_tuple = False)
            adj_index_stop1 = adj_index_stop1.detach().cpu().numpy()

            adj_index_start2 = (edge_index[1] == start_node).nonzero(as_tuple = False)
            adj_index_start2 = adj_index_start2.detach().cpu().numpy()

            adj_index_stop2 = (edge_index[1] == stop_node).nonzero(as_tuple = False)
            adj_index_stop2 = adj_index_stop2.detach().cpu().numpy()

            adj_index = np.concatenate([adj_index_start1, adj_index_stop1, adj_index_start2, adj_index_stop2 ], axis=0)

            idx_reserve = np.delete(idx_reserve, adj_index)

            idx_reserve = idx_reserve.astype(int)
            # index_tensor = torch.from_numpy(idx_reserve)
            edge_attr = data.edge_attr[idx_reserve]

            edge_index_combine = edge_index[:, idx_reserve].clone().detach()
            edge_attr_combine = edge_attr
        else:
            idx_reserve = np.linspace(0, N.clone().detach().cpu() - 1, N.clone().detach().cpu())
            idx_reserve = idx_reserve.astype(int)
            # index_tensor = torch.from_numpy(idx_reserve)
            edge_attr = data.edge_attr[idx_reserve]

            edge_index_combine = edge_index[:, idx_reserve].clone().detach()
            edge_attr_combine = edge_attr

        robust_edge_index = torch.cat([robust_edge_index, edge_index_combine], dim=1)


        robust_edge_attr = torch.cat([robust_edge_attr, edge_attr_combine])

    return robust_edge_index, robust_edge_attr , total_edges

def drop_info_return_edge_removal_according_to_effect(data, effect):
    robust_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    robust_edge_attr = torch.tensor([]).to(data.x.device)

    edge_indices, _, _, num_edges, cum_edges = split_batch(data)
    counter = 0
    effect = effect.transpose(0,1)
    effect = effect * -1
    for edge_index, N, C, effect_graph in zip(edge_indices, num_edges, cum_edges, effect):

        idx_reserve = np.linspace(0, N.clone().detach().cpu() - 1, N.clone().detach().cpu())
        # idx_reserve = np.delete(idx_reserve, index)

        effect_detach = effect_graph.detach().cpu().numpy()
        rank = np.argpartition(-effect_detach, 5)
        index = rank[:5]

        index_keep = torch.nonzero((effect_graph<=0.1))

        # index = np.delete(index, index_keep)


        if effect_graph[index[2]] >= 0.1 :
            idx_reserve = np.delete(idx_reserve, index)

        # start_node = edge_index[0,index]
        # stop_node = edge_index[1,index]
        #
        # adj_index_start1 = (edge_index[0] == start_node).nonzero(as_tuple = False)
        # adj_index_start1 = adj_index_start1.detach().cpu().numpy()
        #
        # adj_index_stop1 = (edge_index[0] == stop_node).nonzero(as_tuple = False)
        # adj_index_stop1 = adj_index_stop1.detach().cpu().numpy()
        #
        # adj_index_start2 = (edge_index[1] == start_node).nonzero(as_tuple = False)
        # adj_index_start2 = adj_index_start2.detach().cpu().numpy()
        #
        # adj_index_stop2 = (edge_index[1] == stop_node).nonzero(as_tuple = False)
        # adj_index_stop2 = adj_index_stop2.detach().cpu().numpy()
        #
        # adj_index = np.concatenate([adj_index_start1, adj_index_stop1, adj_index_start2, adj_index_stop2 ], axis=0)
        #
        # idx_reserve = np.delete(idx_reserve, adj_index)
        #
        # idx_reserve = idx_reserve.astype(int)
        # index_tensor = torch.from_numpy(idx_reserve)
        edge_attr = data.edge_attr[idx_reserve]

        edge_index_combine = edge_index[:, idx_reserve].clone().detach()
        edge_attr_combine = edge_attr

        robust_edge_index = torch.cat([robust_edge_index, edge_index_combine], dim=1)


        robust_edge_attr = torch.cat([robust_edge_attr, edge_attr_combine])

    return robust_edge_index, robust_edge_attr


def drop_info_return_full_score_judge(data, edge_score, ratio):
    robust_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    robust_edge_weight = torch.tensor([]).to(data.x.device)
    robust_edge_attr = torch.tensor([]).to(data.x.device)
    full_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    full_edge_weight = torch.tensor([]).to(data.x.device)
    full_edge_attr = torch.tensor([]).to(data.x.device)

    edge_indices, _, _, num_edges, cum_edges = split_batch(data)

    counter_trigger = 0
    counter_not_trigger = 0

    for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
        # index_reserve = torch.nonzero(edge_score[C:C+N]>=0.3)

        # n_reserve = max(int(ratio * N), index_reserve.shape[0]-1)
        n_reserve = int(0.3 * N)

        if n_reserve != int(ratio * N):
            counter_trigger = counter_trigger + 1
        else:
            counter_not_trigger = counter_not_trigger + 1

        edge_attr = data.edge_attr[C:C+N]
        single_mask = edge_score[C:C+N]
        single_mask_detach = edge_score[C:C+N].detach().cpu().numpy()
        rank = np.argpartition(-single_mask_detach, n_reserve)
        idx_reserve = rank[:n_reserve]

        robust_edge_index = torch.cat([robust_edge_index, edge_index[:, idx_reserve]], dim=1)
        full_edge_index = torch.cat([full_edge_index, edge_index[:, :]], dim=1)

        robust_edge_weight = torch.cat([robust_edge_weight, single_mask[idx_reserve]])
        full_edge_weight = torch.cat([full_edge_weight,  single_mask])

        robust_edge_attr = torch.cat([robust_edge_attr, edge_attr[idx_reserve]])
        full_edge_attr = torch.cat([full_edge_attr, edge_attr])

    p = counter_trigger/(counter_trigger + counter_not_trigger)




    return (robust_edge_index, robust_edge_attr, robust_edge_weight), \
        (full_edge_index, full_edge_attr, full_edge_weight)

def drop_info_return_edge_score_only(data, edge_score, ratio):
    robust_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    robust_edge_weight = torch.tensor([]).to(data.x.device)
    robust_edge_attr = torch.tensor([]).to(data.x.device)
    full_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    full_edge_weight = torch.tensor([]).to(data.x.device)
    full_edge_attr = torch.tensor([]).to(data.x.device)



    edge_indices, _, _, num_edges, cum_edges = split_batch(data)

    robust_edge_index = torch.cat((edge_indices), dim=1).to(data.x.device)
    robust_edge_attr = data.edge_attr.to(data.x.device)
    robust_edge_weight = edge_score.to(data.x.device)

    index = torch.nonzero(robust_edge_weight)

    robust_edge_index = robust_edge_index[:,index]
    robust_edge_attr = robust_edge_attr[index]
    robust_edge_weight = robust_edge_weight[index]

    robust_edge_index = robust_edge_index.squeeze(dim=2)

    full_edge_index = torch.cat((edge_indices), dim=1).to(data.x.device)
    full_edge_attr = data.edge_attr.to(data.x.device)
    full_edge_weight = torch.ones(edge_score.shape[0]).to(data.x.device)


    # for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
    #     n_reserve =  int(ratio * N)
    #     edge_attr = data.edge_attr[C:C+N]
    #     single_mask = edge_score[C:C+N]
    #     single_mask_detach = edge_score[C:C+N].detach().cpu().numpy()
    #     rank = np.argpartition(-single_mask_detach, n_reserve)
    #     idx_reserve = rank[:n_reserve]
    #
    #     robust_edge_index = torch.cat([robust_edge_index, edge_index[:, idx_reserve]], dim=1)
    #     full_edge_index = torch.cat([full_edge_index, edge_index[:, :]], dim=1)
    #
    #     robust_edge_weight = torch.cat([robust_edge_weight, single_mask[idx_reserve]])
    #     full_edge_weight = torch.cat([full_edge_weight,  single_mask])
    #
    #     robust_edge_attr = torch.cat([robust_edge_attr, edge_attr[idx_reserve]])
    #     full_edge_attr = torch.cat([full_edge_attr, edge_attr])




    return (robust_edge_index, robust_edge_attr, robust_edge_weight), \
        (full_edge_index, full_edge_attr, full_edge_weight)



def bool_vec(length, r_True, shuffle=True):
    n_True = math.ceil(length * r_True)
    n_False = length - n_True
    vec = np.concatenate([np.zeros(n_False, dtype=np.bool), np.ones(n_True, dtype=np.bool)])
    if shuffle:
        np.random.shuffle(vec)

    return vec


def sample(dataset, ratio):
    reserve = bool_vec(len(dataset), ratio)
    reserve = torch.tensor(reserve).bool()
    return dataset[reserve]


def relabel(x, edge_index, batch, pos=None):
        
    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos


def relabel_score_drop(x, edge_index, batch, edge_score, pos=None):
    num_nodes = x.size(0)
    index = torch.nonzero(edge_score)
    select_index = torch.cat((edge_index[0,index], edge_index[1,index]), dim=1)
    sub_nodes = torch.unique(select_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos


def get_broken_graph(g, broken_ratio, connectivity=True):

    edge_indices, num_nodes, cum_nodes, num_edges, _ = split_batch(g)
    out_edge_ratio = []
    broken_masks = []
    for edge_index, N, C, E in zip(edge_indices, num_nodes.tolist(),
                                cum_nodes.tolist(), num_edges.tolist()):
        if connectivity:
            flag = 0
            node_idx = np.random.choice([i for i in range(N)])
            node_idx = torch.tensor([node_idx])
            num_edges = int(broken_ratio * E)
            for num_hops in range(1, MAX_DIAM):
                _, _, _, broken_mask = bid_k_hop_subgraph(
                    node_idx=node_idx, 
                    num_hops=num_hops, 
                    edge_index=edge_index-C,
                    num_nodes=N)
                if broken_mask.sum() >= num_edges:
                    flag = 1
                    break
            if flag == 0:
                print("ERROR!")
        else:
            broken_mask = bool_vec(E, r_True=broken_ratio, shuffle=True)
            broken_mask = torch.tensor(broken_mask, dtype=torch.float)
        
        broken_masks.append(broken_mask)
        out_edge_ratio.append(broken_mask.sum().float()/E)
    broken_masks = torch.cat(broken_masks, dim=0).bool()
    broken_edge_index = g.edge_index[:, broken_masks]
    broken_edge_attr = g.edge_attr[broken_masks]
    out_edge_ratio = torch.tensor(out_edge_ratio).to(g.x.device)

    return broken_edge_index, broken_edge_attr, out_edge_ratio


# Bidirectional k-hop subgraph
# adapted from torch-geometric.utils.subgraph
def bid_k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        if len(subsets) > 1:
            node_mask[subsets[-2]] = True
        edge_mask1 = torch.index_select(node_mask, 0, row)
        edge_mask2 = torch.index_select(node_mask, 0, col)
        subsets.append(col[edge_mask1])
        subsets.append(row[edge_mask2])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


def get_syn_ground_truth_graph(g):

    _, _, _, num_edges, cum_edges = split_batch(g)
    
    nodel_label = np.concatenate(g.z, axis=0)
    row, col = g.edge_index.detach().cpu().numpy()
    broken_mask = torch.tensor(nodel_label[row] * nodel_label[col] > 0, dtype=torch.bool)
    broken_edge_index = g.edge_index[:, broken_mask]
    broken_edge_attr = g.edge_attr[broken_mask]
    out_edge_ratio = []
    for E, C in zip(num_edges.tolist(), cum_edges.tolist()):
        out_edge_ratio.append(broken_mask[C: C + E].sum().float()/E)
    
    out_edge_ratio = torch.tensor(out_edge_ratio).to(g.x.device)
    return broken_edge_index, broken_edge_attr, out_edge_ratio


def get_single_ground_truth_graph(g):

    _, _, _, num_edges, cum_edges = split_batch(g)
    nodel_label = np.concatenate(g.z, axis=0)
    row, col = g.edge_index.detach().cpu().numpy()
    broken_mask = torch.tensor(nodel_label[row] * nodel_label[col] > 0, dtype=torch.bool)
    
    broken_edge_indices = torch.LongTensor([[],[]]).to(g.x.device)
    broken_edge_attrs = torch.LongTensor([]).to(g.x.device)
    out_edge_ratio = []
    for E, C in zip(num_edges.tolist(), cum_edges.tolist()):
        edge_idx = torch.nonzero(broken_mask[C: C + E]).view(-1) + C
        edge_index = g.edge_index[:, edge_idx]
        node_idx = np.random.choice(np.unique(edge_index.detach().cpu().numpy()))
        node_idx = torch.tensor([node_idx]).to(g.x.device)
        _, broken_edge_index, _, edge_mask = bid_k_hop_subgraph(node_idx, num_hops=5, edge_index=edge_index)
        broken_edge_attr = g.edge_attr[C: C + E][edge_idx - C][edge_mask]
        broken_edge_indices = torch.cat([broken_edge_indices, broken_edge_index], dim=1)
        broken_edge_attrs = torch.cat([broken_edge_attrs, broken_edge_attr], dim=0)
        out_edge_ratio.append(float(broken_edge_index.size(1)) / E)
        
    out_edge_ratio = torch.tensor(out_edge_ratio).to(g.x.device)
    return broken_edge_indices, broken_edge_attrs, out_edge_ratio


def get_mnist_ground_truth_graph(g):
    
    _, _, _, num_edges, cum_edges = split_batch(g)
    
    nodel_label = torch.tensor(g.x.view(-1) > 0, dtype=torch.bool)
   
    row, col = g.edge_index.detach().cpu().numpy()
    broken_mask = torch.tensor(nodel_label[row] * nodel_label[col] > 0, dtype=torch.bool)
    broken_edge_index = g.edge_index[:, broken_mask]
    broken_edge_attr = g.edge_attr[broken_mask]
    out_edge_ratio = []
    for E, C in zip(num_edges.tolist(), cum_edges.tolist()):
        out_edge_ratio.append(broken_mask[C: C + E].sum().float()/E)
    
    out_edge_ratio = torch.tensor(out_edge_ratio).to(g.x.device)
    return broken_edge_index, broken_edge_attr, out_edge_ratio


def get_ground_truth_graph(args, g):
    if args.dataset == 'ba3':
        return get_single_ground_truth_graph(g)
    elif args.dataset == 'tr3':
        return get_syn_ground_truth_graph(g)
    elif args.dataset == 'mnist':
        return get_mnist_ground_truth_graph(g)