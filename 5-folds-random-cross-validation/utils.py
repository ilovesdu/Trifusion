import numpy as np
import torch
from sklearn import metrics
import dgl
import networkx as nx
def caculate_metrics(real_score, pre_score):
    y_true = real_score
    y_pre = pre_score
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pre, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision_u, recall_u, thresholds_u = metrics.precision_recall_curve(y_true, y_pre)
    aupr = metrics.auc(recall_u, precision_u)

    y_score = [0 if j < 0.5 else 1 for j in y_pre]

    acc = metrics.accuracy_score(y_true, y_score)
    f1 = metrics.f1_score(y_true, y_score)
    recall = metrics.recall_score(y_true, y_score)
    precision = metrics.precision_score(y_true, y_score)

    metric_result = [auc, aupr, acc, f1, precision, recall]
    return metric_result

def integ_similarity(M1, M2):
    for i in range(len(M1)):
        for j in range(len(M1)):
            if M1[i][j] == 0:
                M1[i][j] = M2[i][j]
    return M1


def get_graph_adj(matrix, device):
    graph_adj = []
    for i in range(matrix.shape[0]):
        temp_adj = []
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                temp_adj.append(1)
            else:
                temp_adj.append(0)
        graph_adj.append(temp_adj)
    graph_adj = np.array(graph_adj).reshape(matrix.shape[0], matrix.shape[1])
    return torch.tensor(graph_adj, device=device).to(torch.float32)


def topk_filtering(args, d_d, k: int):
    d_d = d_d.numpy()
    for i in range(len(d_d)):
        sorted_idx = np.argpartition(d_d[i], -k - 1)
        d_d[i, sorted_idx[-k - 1:-1]] = 1
    return torch.tensor(np.where(d_d == 1), device=args.device)

def get_edge_index(matrix, device):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.tensor(edge_index, dtype=torch.long, device=device)

def k_matrix(matrix, k=40):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)






def create_dgl_graph(matrix1, matrix2, matrix3, num_nodes=1444):
    """
    Concatenates three 2xN matrices (or tensors) column-wise and creates a DGL graph.

    Parameters:
    - matrix1, matrix2, matrix3: Three 2xN NumPy arrays or PyTorch tensors.
    - num_nodes: The number of nodes in the graph (default: 1444).

    Returns:
    - A DGL graph created from the concatenated matrices.
    """
    # 检查输入类型，如果是张量，则转换为NumPy数组
    if torch.is_tensor(matrix1):
        matrix1 = matrix1.detach().cpu().numpy()
    if torch.is_tensor(matrix2):
        matrix2 = matrix2.detach().cpu().numpy()
    if torch.is_tensor(matrix3):
        matrix3 = matrix3.detach().cpu().numpy()

    # 确保输入矩阵维度正确
    assert matrix1.shape[0] == 2 and matrix2.shape[0] == 2 and matrix3.shape[0] == 2, "All matrices must be 2xN."

    # 拼接矩阵
    concatenated_matrix = np.concatenate([matrix1, matrix2, matrix3], axis=1)

    # 创建边
    src_nodes = np.concatenate([concatenated_matrix[0, :], concatenated_matrix[1, :]])
    dst_nodes = np.concatenate([concatenated_matrix[1, :], concatenated_matrix[0, :]])

    # 创建DGL图
    graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
    return graph




def compute_clustering_coefficients_multigraph(g):
    """
    计算DGL多图中每个节点的聚类系数。

    参数:
    g (dgl.DGLGraph): 输入的DGL多图。

    返回:
    torch.Tensor: 节点的聚类系数。
    """
    # 将DGL多图转换为NetworkX多图
    nx_multigraph = g.to_networkx().to_undirected()

    # 将多图转换为简单图
    nx_simple_graph = nx.Graph(nx_multigraph)

    # 计算聚类系数
    clustering_coeffs = nx.clustering(nx_simple_graph)

    # 将聚类系数存储到张量中
    clustering_coeffs_tensor = torch.zeros(len(clustering_coeffs))
    for node, coeff in clustering_coeffs.items():
        clustering_coeffs_tensor[node] = coeff

    return clustering_coeffs_tensor

def build_heterograph(circrna_disease_matrix, circSimi, disSimi):

    # 求出相似矩阵平均值，大于平均值，认为有连边
    # rna 0.4055
    # mean_Similarity_circRNA = np.mean(circSimi)
    # 3077*3077
    matAdj_circ = np.where(circSimi > 0.5, 1, 0)
    #print(np.shape(matAdj_circ))
    # dis 0.0903
    # mean_Similarity_disease = np.mean(disSimi)
    # 313*313
    matAdj_dis = np.where(disSimi > 0.5, 1, 0)
    #print(np.shape(matAdj_dis))
    # Heterogeneous adjacency matrix
    # np.hstack()：按水平方向（列顺序）堆叠数组构成一个新的数组堆叠的数组需要具有相同的维度
    # 3077*3390
    h_adjmat_1 = np.hstack((matAdj_circ, circrna_disease_matrix))
    # 313*3390
    h_adjmat_2 = np.hstack((circrna_disease_matrix.transpose(), matAdj_dis))
    # np.vstack()：按垂直方向（行顺序）堆叠数组构成一个新的数组堆叠的数组需要具有相同的维度
    # 3390*3390
    Heterogeneous1 = np.vstack((h_adjmat_1, h_adjmat_2))
    # heterograph
    gcd = dgl.heterograph(
        data_dict={
            # Heterogeneous.nonzero()返回非0元素索引
            ('circRNA_disease', 'interaction', 'circRNA_disease'): Heterogeneous1.nonzero()},
        num_nodes_dict={
            'circRNA_disease': 1444
        })

    return gcd

def create_matrix_from_indices(indices, shape=(853, 591)):
    # 创建一个全为0的矩阵
    matrix = np.zeros(shape, dtype=int)

    # 确保indices是2xN的
    assert indices.shape[0] == 2, "Indices array must be of shape 2xN"

    # 将相应的位置设置为1
    for col, row in zip(indices[0], indices[1]):
        if 0 <= col < shape[0] and 0 <= row < shape[1]:
            matrix[col, row] = 1
        else:
            print(f"Warning: Index out of bounds - ({col}, {row})")

    return matrix



