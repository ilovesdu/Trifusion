import os
from sklearn.model_selection import KFold
from param import *
from utils import *

args = parse_args()


def loading_similarity_feature(args):
    path = args.path
    device = args.device

    similarity_feature = {}

    "MiRNA Sequence Similarity"
    m_seq_sim = np.loadtxt(os.path.join(path, 'm_ss.csv'), delimiter=',', dtype=float)
    m_seq_sim = torch.tensor(m_seq_sim, device=device).to(torch.float32)
    m_ss_edge_idx = get_edge_index(m_seq_sim, device)
    m_ss_graph = dgl.graph((m_ss_edge_idx[0], m_ss_edge_idx[1]))
    similarity_feature['m_ss'] = {'Data': m_seq_sim, 'Edge': m_ss_edge_idx, 'Graph': m_ss_graph}

    "MiRNA Gaussian Similarity"
    m_gauss_sim = np.loadtxt(os.path.join(path, 'm_gs.csv'), delimiter=',', dtype=float)
    m_gauss_sim = torch.tensor(m_gauss_sim, device=device).to(torch.float32)
    m_gs_edge_idx = get_edge_index(m_gauss_sim, device)
    m_gs_graph = dgl.graph((m_gs_edge_idx[0], m_gs_edge_idx[1]))
    similarity_feature['m_gs'] = {'Data': m_gauss_sim, 'Edge': m_gs_edge_idx, 'Graph': m_gs_graph}

    "MiRNA Function Similarity"
    m_func_sim = np.loadtxt(os.path.join(path, 'm_fs.csv'), delimiter=',', dtype=float)
    m_func_sim = torch.tensor(m_func_sim, device=device).to(torch.float32)
    m_fs_edge_idx = get_edge_index(m_func_sim, device)
    m_fs_graph = dgl.graph((m_fs_edge_idx[0], m_fs_edge_idx[1]))
    similarity_feature['m_fs'] = {'Data': m_func_sim, 'Edge': m_fs_edge_idx, 'Graph': m_fs_graph}

    "Disease Semantic Similarity"
    d_sem_sim = np.loadtxt(os.path.join(path, 'd_ss.csv'), delimiter=',', dtype=float)
    d_sem_sim = torch.tensor(d_sem_sim, device=device).to(torch.float32)
    d_ss_edge_idx = get_edge_index(d_sem_sim, device)
    d_ss_graph = dgl.graph((d_ss_edge_idx[0], d_ss_edge_idx[1]))
    similarity_feature['d_ss'] = {'Data': d_sem_sim, 'Edge': d_ss_edge_idx, 'Graph': d_ss_graph}

    "Disease Gaussian Similarity"
    d_gauss_sim = np.loadtxt(os.path.join(path, 'd_gs.csv'), delimiter=',', dtype=float)
    d_gauss_sim = torch.tensor(d_gauss_sim, device=device).to(torch.float32)
    d_gs_edge_idx = get_edge_index(d_gauss_sim, device)
    d_gs_graph = dgl.graph((d_gs_edge_idx[0], d_gs_edge_idx[1]))
    similarity_feature['d_gs'] = {'Data': d_gauss_sim, 'Edge': d_gs_edge_idx, 'Graph': d_gs_graph}

    miRNA_similarity = (m_gauss_sim + m_seq_sim) / 2
    similarity_feature['m_s'] = {'Data': miRNA_similarity}
    disease_similarity = (d_sem_sim + d_gauss_sim) / 2
    similarity_feature['d_s'] = {'Data': disease_similarity}

    return similarity_feature


def data_preprocess(args):
    path = args.path
    k_fold = args.kfolds
    seed = args.SEED
    edge_idx_dict = dict()
    my_graph = dict()

    md_matrix = np.loadtxt(os.path.join(path, 'm_d.csv'), delimiter=',', dtype=int)
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=1)
    row_idx = np.random.choice(range(args.numrna), size=args.numrna, replace=False)
    row_train_idx, row_valid_idx = [], []
    for row_train_index, row_valid_index in kf.split(row_idx):
        row_train_idx.append(row_train_index)
        row_valid_idx.append(row_valid_index)

    for k in range(k_fold):
        # Valid samples
        pos_samples_valid = (np.array([], dtype=int), np.array([], dtype=int))
        for i in row_valid_idx[k]:
            pos = np.where(md_matrix[i, :] == 1)
            pos = (np.full(pos[0].shape, i), pos[0] + md_matrix.shape[0])
            pos_samples_valid = (np.concatenate([pos_samples_valid[0], pos[0]]),
                                 np.concatenate([pos_samples_valid[1], pos[1]]))
        rng = np.random.default_rng(seed=seed)
        pos_samples_valid_shuffled = rng.permutation(pos_samples_valid, axis=1)
        valid_pos_samples = pos_samples_valid_shuffled

        neg_samples_valid = (np.array([], dtype=int), np.array([], dtype=int))
        for i in row_valid_idx[k]:
            neg = np.where(md_matrix[i, :] == 0)
            neg = (np.full(neg[0].shape, i), neg[0] + md_matrix.shape[0])
            neg_samples_valid = (np.concatenate([neg_samples_valid[0], neg[0]]),
                                 np.concatenate([neg_samples_valid[1], neg[1]]))
        neg_samples_valid_shuffled = rng.permutation(neg_samples_valid, axis=1)[:, :pos_samples_valid_shuffled.shape[1]]
        valid_neg_samples = neg_samples_valid_shuffled

        # Train samples
        pos_samples_train = (np.array([], dtype=int), np.array([], dtype=int))
        for i in row_train_idx[k]:
            pos = np.where(md_matrix[i, :] == 1)
            pos = (np.full(pos[0].shape, i), pos[0] + md_matrix.shape[0])
            pos_samples_train = (np.concatenate([pos_samples_train[0], pos[0]]),
                                 np.concatenate([pos_samples_train[1], pos[1]]))
        rng = np.random.default_rng(seed=seed)
        pos_samples_train_shuffled = rng.permutation(pos_samples_train, axis=1)
        train_pos_samples = pos_samples_train_shuffled

        neg_samples_train = (np.array([], dtype=int), np.array([], dtype=int))
        for i in row_train_idx[k]:
            neg = np.where(md_matrix[i, :] == 0)
            neg = (np.full(neg[0].shape, i), neg[0] + md_matrix.shape[0])
            neg_samples_train = (np.concatenate([neg_samples_train[0], neg[0]]),
                                 np.concatenate([neg_samples_train[1], neg[1]]))
        neg_samples_train_shuffled = rng.permutation(neg_samples_train, axis=1)[:, :pos_samples_train_shuffled.shape[1]]
        train_neg_samples = neg_samples_train_shuffled

        edge_idx_dict[str(k)] = {}
        my_graph[str(k)] = {}

        fold_train_pos80 = train_pos_samples
        fold_valid_pos20 = valid_pos_samples
        fold_train_neg80 = train_neg_samples
        fold_valid_neg20 = valid_neg_samples

        fold_100p_100n = np.hstack((np.hstack((fold_train_pos80, fold_valid_pos20)),
                                    np.hstack((fold_train_neg80, fold_valid_neg20))))
        fold_train_80p_80n = np.hstack((fold_train_pos80, fold_train_neg80))
        fold_train_label_80p_80n = np.hstack((np.ones(fold_train_pos80.shape[1]),
                                              np.zeros(fold_train_neg80.shape[1])))
        fold_valid_20p_20n = np.hstack((fold_valid_pos20, fold_valid_neg20))
        fold_valid_label_20p_20n = np.hstack((np.ones(fold_valid_pos20.shape[1]),
                                              np.zeros(fold_valid_neg20.shape[1])))

        edge_idx_dict[str(k)]["all_edge"] = torch.tensor(fold_100p_100n).to(torch.long).to(device=args.device)
        my_graph[str(k)]["all_graph"] = dgl.graph((fold_100p_100n[0], fold_100p_100n[1])).to(device=args.device)

        edge_idx_dict[str(k)]["fold_train_edges_80p_80n"] = torch.tensor(fold_train_80p_80n).to(torch.long).to(
            device=args.device)
        my_graph[str(k)]["fold_train_edges_80p_80n"] = dgl.graph((fold_train_80p_80n[0],
                                                                  fold_train_80p_80n[1])).to(device=args.device)

        my_graph[str(k)]['fold_train_pos'] = dgl.graph((fold_train_pos80[0],
                                                        fold_train_pos80[1])).to(device=args.device)

        edge_idx_dict[str(k)]["fold_train_label_80p_80n"] = torch.tensor(fold_train_label_80p_80n).to(torch.float32).to(
            device=args.device)
        edge_idx_dict[str(k)]["fold_valid_label_20p_20n"] = torch.tensor(fold_valid_label_20p_20n).to(torch.float32).to(
            device=args.device)

        edge_idx_dict[str(k)]["fold_valid_edges_20p_20n"] = torch.tensor(fold_valid_20p_20n).to(torch.long).to(
            device=args.device)
        my_graph[str(k)]["fold_valid_edges_20p_20n"] = dgl.graph((fold_valid_20p_20n[0],
                                                                  fold_valid_20p_20n[1])).to(device=args.device)

    return edge_idx_dict, my_graph
