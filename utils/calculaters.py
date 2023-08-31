import torch


def calculate_percentages(tensor, percent):

    percent = percent / 4

    # 按照全部元素大小进行排序
    sorted_tensor, _ = torch.sort(tensor.view(-1))

    # 选出按照全部元素大小的后20%处的元素
    num_elements = sorted_tensor.numel()
    num_selected = int(num_elements * percent)
    check_element = sorted_tensor[num_selected].item()

    # 将tensor按照第一维度拆分成不同的元素组
    element_groups = torch.split(tensor, 1, dim=1)

    # 计算元素组中值比check_element小的元素数占总数的百分比
    percentages = []
    for group in element_groups:
        smaller_elements = torch.sum(group < check_element).item()
        percentage = (smaller_elements / group.size(0)) * 100
        percentages.append(percentage)

    return percentages


def swap_dimensions(tensor):
    # 交换张量的维度0和维度1
    return tensor.transpose(0, 1)


def find_matching_elements(robust_edge_index_pre, robust_edge_index, robust_edge_weight):
    # 将robust_edge_index_pre的维度0和维度1交换
    robust_edge_index_pre_swapped = swap_dimensions(robust_edge_index_pre)
    # 将robust_edge_index的维度1和维度0交换
    robust_edge_index_swapped = swap_dimensions(robust_edge_index)

    match_index = []
    for i in range(robust_edge_index_swapped.size(0)):

        edge = robust_edge_index_swapped[i]
        assert edge[0] <= edge[1]

        matching_indices = torch.nonzero(robust_edge_index_pre_swapped[:, 0] == edge[0]).squeeze()

        if matching_indices.dim() == 0:
            matching_indices_list = [int(matching_indices.item())]
        else:
            matching_indices_list = matching_indices.tolist()


        for j in matching_indices_list:
            target = robust_edge_index_pre_swapped[j]
            assert target[0] <= target[1]
            if torch.equal(edge, target):
                match_index.append(i)
                break

    # 根据索引选出对应的元素
    selected_edge_index = robust_edge_index[:,match_index]
    selected_edge_weight = robust_edge_weight[match_index]

    return selected_edge_index, selected_edge_weight


