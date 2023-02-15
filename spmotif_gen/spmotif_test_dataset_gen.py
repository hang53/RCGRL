from BA3_loc import *
from tqdm import tqdm
import os.path as osp
import warnings
warnings.filterwarnings("ignore")

global_b = '0.9' # Set bias degree here
data_dir = f'../data/SPMotif-{global_b}/raw/'
os.makedirs(data_dir, exist_ok=True)

def get_house(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph:

    Start with a tree and attach HOUSE-shaped subgraphs.
    """
    list_shapes = [["house"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name

def get_cycle(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph:

    Start with a tree and attach cycle-shaped (directed edges) subgraphs.
    """
    list_shapes = [["dircycle"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name

def get_crane(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph:

    Start with a tree and attach crane-shaped subgraphs.
    """
    list_shapes = [["crane"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name

# no bias for test dataset
edge_index_list, label_list = [], []
ground_truth_list, role_id_list, pos_list = [], [], []

def graph_stats_large(base_num):
    if base_num == 1:
        base = 'tree'
        width_basis=np.random.choice(range(3,6))
    if base_num == 2:
        base = 'ladder'
        width_basis=np.random.choice(range(30,50))
    if base_num == 3:
        base = 'wheel'
        width_basis=np.random.choice(range(60,80))
    return base, width_basis

e_mean, n_mean = [], []
for _ in tqdm(range(2000)):
    base_num = np.random.choice([1,2,3]) # uniform
    base, width_basis = graph_stats_large(base_num)

    G, role_id, name = get_cycle(basis_type=base, nb_shapes=1,
                                    width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(0)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), np.mean(n_mean), np.mean(e_mean)))

e_mean, n_mean = [], []
for _ in tqdm(range(2000)):
    base_num = np.random.choice([1,2,3])
    base, width_basis = graph_stats_large(base_num)

    G, role_id, name = get_house(basis_type=base, nb_shapes=1,
                                    width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(1)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), np.mean(n_mean), np.mean(e_mean)))

e_mean, n_mean = [], []
for _ in tqdm(range(2000)):
    base_num = np.random.choice([1,2,3])
    base, width_basis = graph_stats_large(base_num)

    G, role_id, name = get_crane(basis_type=base, nb_shapes=1,
                                    width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), np.mean(n_mean), np.mean(e_mean)))
np.save(osp.join(data_dir, 'test.npy'), (edge_index_list, label_list, ground_truth_list, role_id_list, pos_list))
