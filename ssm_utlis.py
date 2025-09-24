
import torch
import numpy as np
import pandas as pd
import random
import copy

def print_eval_metrics(results, prefix=""):
    """
    results: dict returned from link_prediction_eval
    prefix: string to prepend (e.g. 'VAL' or 'TEST')
    """
    for split in ['transductive', 'inductive']:
        print(f"\n{prefix} {split.upper()} metrics:")
        for neg_type in ['rnd', 'hst', 'ind']:
            auc = results[split][neg_type]['AUC']
            ap = results[split][neg_type]['AP']
            print(f"  {neg_type.upper():>5} - AUC: {auc:.4f} | AP: {ap:.4f}")

def aggregate_metrics(results_list, split_name):
    # results_list is list of dicts: each dict like {'rnd': {'AUC':.., 'AP':..}, ...}
    metrics = {'rnd': {'AUC': [], 'AP': []}, 'hst': {'AUC': [], 'AP': []}, 'ind': {'AUC': [], 'AP': []}}

    for res in results_list:
        for neg_type in ['rnd', 'hst', 'ind']:
            metrics[neg_type]['AUC'].append(res[neg_type]['AUC'])
            metrics[neg_type]['AP'].append(res[neg_type]['AP'])

    # Compute mean ± std per neg_type and metric
    summary = {}
    for neg_type in metrics:
        summary[neg_type] = {
            'AUC_mean': np.mean(metrics[neg_type]['AUC']),
            'AUC_std': np.std(metrics[neg_type]['AUC']),
            'AP_mean': np.mean(metrics[neg_type]['AP']),
            'AP_std': np.std(metrics[neg_type]['AP']),
        }
    return summary


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0.001, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.best_state_dict = None

    def __call__(self, val_loss, model):
        score = val_loss  # lower loss = higher score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score* 1 + self.delta :
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def combine_sort(dataset, new_dataset):
    """
    Merge old and new edge lists, sort by time,
    and return (src, dst, time, mask).
    
    mask[i] = 0 if edge came from old set,
              1 if edge came from new set
    """

    combined_dataset = copy.deepcopy(dataset)

    # Masks
    old_mask = np.zeros_like(dataset.node_interact_times, dtype=bool)  # 0 = old
    new_mask = np.ones_like(new_dataset.node_interact_times, dtype=bool)   # 1 = new

    # Concatenate
    src = np.concatenate([dataset.src_node_ids, new_dataset.src_node_ids])
    dst = np.concatenate([dataset.dst_node_ids, new_dataset.dst_node_ids])
    time = np.concatenate([dataset.node_interact_times, new_dataset.node_interact_times])
    mask = np.concatenate([old_mask, new_mask])
    edge_ids = np.concatenate([dataset.edge_ids, new_dataset.edge_ids])

    # Sort by time
    idx = np.argsort(time)

    combined_dataset.src_node_ids = src[idx]
    combined_dataset.dst_node_ids = dst[idx]
    combined_dataset.node_interact_times = time[idx]
    combined_dataset.inductive_mask = mask[idx]
    combined_dataset.edge_ids= edge_ids[idx]
    return combined_dataset
    

def build_laplacian_windowed_agg(src, dst, num_nodes,weight, device):
    try:
        edge_index = torch.stack([src, dst], dim=0)
    except RuntimeError:
        edge_index = torch.stack([src[None,:], dst[None,:]], dim=0)
    # torch.ones(src.size(0))
    if weight== None:
        weight = torch.ones(src.size(0))
    A = torch.sparse_coo_tensor(edge_index, weight, (num_nodes, num_nodes), device=device).to_dense()

    # symmetrize and binarize adjacency
    A = ((A + A.T) > 0).float()

    # degree inverse
    D = torch.diag(torch.clamp_min(A.sum(dim=1),1)**(-0.5))

    # Laplacian: L = I - D^{-1} A
    L = torch.eye(num_nodes, device=device) - D@A@D
    return L

def build_laplacian_windowed(tau,T,src,dst,src_t,dst_t,active_ids,sub_src, sub_dst, weight=None, device='cpu'):
    if weight is None:
        weight = torch.ones(src.size(0), device=device)

    # Build adjacency matrix (dense once)
    # edge_index = torch.stack([sub_src, sub_dst], dim=0)
    # A = torch.sparse_coo_tensor(edge_index, weight, (num_nodes, num_nodes), device=device).to_dense()

    
    # # Degree vector
    # D = torch.sum(A, dim=1)
    # D[D == 0] = 1.0
    # D_inv = 1.0 / D

    # Normalized Laplacian: L = I - D^-1 * A
    # Multiply each row of A by D_inv
    L = torch.eye(len(active_ids), device=device) 

    src_tau = src[:tau]
    dst_tau = dst[:tau]
    src_taup = src[:tau+T]
    dst_taup = dst[:tau+T]
    # Delta_L:
    delta_L = torch.zeros_like(L,device=device)
    for i,node in enumerate(active_ids):
        new_src = src_t[dst_t==node]
        if len(new_src)==0:
            continue
        new_src_ids = sub_src[dst_t==node]
        old_nu = src_tau[dst_tau==node]
        du = old_nu.numel()
        if du == 0:
            du = 1

        new_nu = src_taup[dst_taup==node]
        du_ = new_nu.numel()
        if du_ ==0:
            du_ = 1

        old_w = torch.hstack([torch.sum(old_nu==src_x)/du for src_x in new_src]) 
        new_w = torch.hstack([torch.sum(new_nu==src_x)/du_ for src_x in new_src]) 
        delta_L[i,new_src_ids] = old_w-new_w
        L[i,new_src_ids] = -new_w
    return L, delta_L

def positional_encoding_mixer(timestamps, num_dims):
    alpha = num_dims**(0.5)
    beta  = num_dims**(0.5)
    i = np.arange(num_dims)[None]
    omega_raise = alpha**((-i+1)/beta)
    pos = timestamps[:,None]  # [T, 1]
    
    return np.cos(pos * omega_raise)

# def aggregate_edge_features_to_nodes(src, dst, edge_feats, num_nodes):
#     x = torch.zeros(num_nodes, edge_feats.shape[1], device=edge_feats.device)
#     for i in range(len(edge_feats)):
#         x[src[i]] += edge_feats[i]
#         x[dst[i]] += edge_feats[i]
#     return x

# def aggregate_edge_features_to_nodes(src, dst, edge_feats, num_nodes):
#     x = torch.zeros(num_nodes, edge_feats.shape[1], device=edge_feats.device)
#     x.index_add_(0, src, edge_feats)  # add edge features to source nodes
#     x.index_add_(0, dst, edge_feats)  # add edge features to destination nodes
#     return x

def aggregate_edge_features_to_nodes(src, dst, edge_feats, num_nodes):
    """
    Vectorized aggregation of edge features to nodes, with zero-padding if edge_feats is shorter than src/dst.
    Args:
        src (LongTensor): source node indices of edges, shape [num_edges_total]
        dst (LongTensor): destination node indices of edges, shape [num_edges_total]
        edge_feats (FloatTensor): edge features, shape [num_edges_original, feat_dim]
        num_nodes (int): number of nodes in graph

    Returns:
        x (FloatTensor): node features aggregated from edges, shape [num_nodes, feat_dim]
    """
    num_edges_total = len(src)
    feat_dim = edge_feats.shape[1]

    # Pad edge_feats with zeros if there are missing edges
    if edge_feats.shape[0] < num_edges_total:
        pad_len = num_edges_total - edge_feats.shape[0]
        pad = torch.zeros(pad_len, feat_dim, device=edge_feats.device, dtype=edge_feats.dtype)
        edge_feats = torch.cat([edge_feats, pad], dim=0)

    # Preallocate node features
    x = torch.zeros(num_nodes, feat_dim, device=edge_feats.device, dtype=edge_feats.dtype)

    # Vectorized addition
    x.index_add_(0, src, edge_feats)
    x.index_add_(0, dst, edge_feats)

    return x

def dir_causal_laplacain(src,dst,delta=1,device='cpu'):
    num_nodes = len(src)*2
    num_src = len(src)
    A = torch.zeros((num_nodes,num_nodes)).to(device)
    for t,node in enumerate(src):
        nindex = num_src+torch.arange(len(src[:t+1]))
        n_mask = nindex[src[:t+1]==node]
        A[t,n_mask] = torch.ones_like(n_mask,device=device,dtype=torch.float32) #torch.exp(-(t-src_mask)/delta)

    for t,node in enumerate(dst):
        #sindex = torch.arange(len(dst[:t+1]))
        nindex = torch.arange(len(dst[:t+1]))
        n_mask = nindex[dst[:t+1]==node]
        A[t+num_src,n_mask] = torch.ones_like(n_mask,device=device,dtype=torch.float32) #torch.exp(-(t-src_mask)/delta)


    A += torch.eye(len(A)).to(device)
    D = torch.sum(A,dim=1)[:,None]
    L = torch.eye(len(A)).to(device)-(1/D)*A

    return L

# def node_unique_batch(src, dst, T, t_start):
#     """
#     src, dst : LongTensor [num_edges]
#     T        : max window size
#     t_start  : current start index
#     """
#     # Take candidate window of size T
#     end_idx = min(t_start + T, src.size(0))
#     src_win = src[t_start:end_idx]
#     dst_win = dst[t_start:end_idx]

#     # Stack edges into node sequence [2, window_size]
#     nodes = torch.cat([src_win, dst_win])

#     # Find first repeat
#     # index of first duplicate == cutoff point
#     _, inv, counts = torch.unique(nodes, return_inverse=True, return_counts=True)
#     repeat_mask = counts[inv] > 1
#     if repeat_mask.any():
#         # position of first repeat in nodes
#         cutoff = repeat_mask.nonzero(as_tuple=False)[0, 0] // 2
#         cutoff = max(1, cutoff)  

#         end_idx = t_start + cutoff
#         src_win = src[t_start:end_idx]
#         dst_win = dst[t_start:end_idx]

#     return src_win, dst_win, end_idx

def node_unique_batch(src, dst, T, t_start):
    """
    src, dst : LongTensor [num_edges]
    T        : max window size
    t_start  : current start index
    """
    # Take candidate window of size T
    end_idx = min(t_start + T, src.size(0))
    src_win = src[t_start:end_idx]
    dst_win = dst[t_start:end_idx]

    # Track seen nodes incrementally (set-based, fast)
    seen = set()
    cutoff = None
    for i, (u, v) in enumerate(zip(src_win.tolist(), dst_win.tolist())):
        if u in seen or v in seen:
            cutoff = i  # first repeat-causing edge
            break
        seen.add(u)
        seen.add(v)

    if cutoff is not None:
        end_idx = t_start + cutoff
        src_win = src[t_start:end_idx]
        dst_win = dst[t_start:end_idx]

    return src_win, dst_win, end_idx

def sample_uniform_fast(nodes, past_self, past_neighbour, k,mode='rnd'):
    node_neighbours = []
    node_neighbours_nodes = []
    
    for node in nodes:
        neighbors = past_neighbour[past_self == node]
        n = neighbors.numel()
        
        if n == 0:
            continue  # no neighbors, just skip
        
        if n <= k:
            sampled = neighbors
        else:
            if mode == 'rnd':
                idx = torch.randint(0, n, (k,), device=neighbors.device)
                sampled = neighbors[idx]
            elif mode== 'rcn':
                sampled = neighbors[:k]
        
        node_neighbours.append(sampled)
        node_neighbours_nodes.append(node.repeat(sampled.numel()))
    
    if len(node_neighbours) == 0:
        return torch.empty(0, dtype=torch.long, device=past_self.device), \
               torch.empty(0, dtype=torch.long, device=past_self.device)
    
    return torch.hstack(node_neighbours), torch.hstack(node_neighbours_nodes)


def NeighbourSampler(src_t,dst_t,neg_dst,src,dst,t_start,tau=1,max_node=20,mode='CW'):
    if mode=='CW':
        if t_start>tau:
            dst_tau = dst[t_start-tau:t_start]
            src_tau = src[t_start-tau:t_start]
        else : 
            dst_tau = dst[:t_start]
            src_tau = src[:t_start]

        src_neighbours,src_neighbours_src  = sample_uniform_fast(src_t,src_tau,dst_tau,max_node) 
        dst_neighbours,dst_neighbours_dst =sample_uniform_fast(dst_t,dst_tau,src_tau,max_node) 
        neg_neighbour,neg_neighbour_dst = sample_uniform_fast(neg_dst,dst_tau,src_tau,max_node)
        
    elif mode=='FX':
        dst_tau = dst[:t_start]
        src_tau = src[:t_start]

        src_neighbours,src_neighbours_src  = sample_uniform_fast(src_t,src_tau,dst_tau,max_node,'rcn') 
        dst_neighbours,dst_neighbours_dst =sample_uniform_fast(dst_t,dst_tau,src_tau,max_node,'rcn') 
        neg_neighbour,neg_neighbour_dst = sample_uniform_fast(neg_dst,dst_tau,src_tau,max_node,'rcn')
    
    return src_neighbours,src_neighbours_src,dst_neighbours,dst_neighbours_dst,neg_neighbour,neg_neighbour_dst


def add_edge_features(src,dst,t,edge_raw_features,time_encoder,time_enc_dim_data):
    df = pd.DataFrame({
        'src':src,
        'dst':dst,
        't': t
    })
    
    # df = df.sort_values("t").reset_index(drop=True)
    df["edge"] = list(zip(df["src"], df["dst"]))
    # f1: cumulative count of (u,v)
    df["f1"] = df.groupby("edge").cumcount()

    # f2: time since last occurrence
    df["last_time"] = df.groupby("edge")["t"].shift()
    df["f2"] = df["t"] - df["last_time"]
    df["f2"] = df["f2"].fillna(1e+11)   # mark first occurrence
    df['f2'] = np.log1p(df['f2'])

    df = df.drop(columns=["last_time"])

    # time_enc = time_encoder(t,time_enc_dim_data)
    time_enc_delta = time_encoder(df['f2'].to_numpy(),time_enc_dim_data)
    # final_edge_feat = np.hstack([edge_raw_features,time_enc,time_enc_delta,df['f1'].to_numpy()[:,None]])
    final_edge_feat = np.hstack([edge_raw_features,time_enc_delta,df['f1'].to_numpy()[:,None]])
    
    return final_edge_feat, df['f2'].to_numpy()


#%% Rapid Neighbour Sampling

# def build_neighbor_index_torch(df, num_nodes=None, device="cpu"):
#     """
#     Build neighbor adjacency in contiguous torch tensors with offsets.
#     Each node's neighbors are stored in a single tensor slice:
#         neighbors[offsets[i]:offsets[i+1]]
#         times[offsets[i]:offsets[i+1]]
    
#     Args:
#         df : pandas.DataFrame with columns [src, dst, t]
#         num_nodes : optional, number of nodes (if None → inferred)
#         device : "cpu" or "cuda"
    
#     Returns:
#         neighbors: torch.LongTensor [E_total]
#         times: torch.LongTensor [E_total]
#         offsets: torch.LongTensor [num_nodes+1]
#     """
#     # infer num_nodes
#     if num_nodes is None:
#         num_nodes = int(max(df["src"].max(), df["dst"].max())) + 1

#     # adjacency list as python first
#     adj = [[] for _ in range(num_nodes)]
#     for src, dst, t in df.itertuples(index=False):
#         adj[src].append((t, dst))
#         adj[dst].append((t, src))

#     # sort and flatten
#     neighbors_list, times_list, offsets = [], [], [0]
#     for node in range(num_nodes):
#         nbrs = sorted(adj[node], key=lambda x: x[0])  # sort by time
#         if nbrs:
#             tvals, nbrs_ids = zip(*nbrs)
#             neighbors_list.extend(nbrs_ids)
#             times_list.extend(tvals)
#         offsets.append(len(neighbors_list))

#     # convert to torch tensors
#     neighbors = torch.tensor(neighbors_list, dtype=torch.long, device=device)
#     times = torch.tensor(times_list, dtype=torch.long, device=device)
#     offsets = torch.tensor(offsets, dtype=torch.long, device=device)

#     return neighbors, times, offsets

import torch

def build_neighbor_index_torch(src, dst, t, num_nodes=None, device="cpu"):
    """
    Build neighbor adjacency in contiguous torch tensors with offsets.
    Assumes (src, dst, t) are already sorted by time.

    Args:
        src, dst : LongTensors [E]
        t        : LongTensor [E]
        num_nodes : optional, number of nodes (if None → inferred)
        device : "cpu" or "cuda"

    Returns:
        neighbors: LongTensor [2*E]
        times: LongTensor [2*E]
        offsets: LongTensor [num_nodes+1]
    """
    if num_nodes is None:
        num_nodes = int(torch.max(torch.cat([src, dst])).item()) + 1

    # undirected expansion
    neighbors = torch.cat([dst, src])
    times     = torch.cat([t, t])
    nodes     = torch.cat([src, dst])

    # just count how many per node → offsets
    counts = torch.bincount(nodes, minlength=num_nodes)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long,device=device), counts.cumsum(0)])

    return neighbors.to(device), times.to(device), offsets.to(device)


def temporal_sample_neighbors(src_nodes, neighbors, times, offsets, tau, k, replace=False):
    """
    Vectorized temporal neighbor sampler (no Python for loop).

    src_nodes: (B,) LongTensor, batch of nodes
    neighbors: (E,) LongTensor, all neighbors (CSR)
    times:     (E,) LongTensor, timestamps aligned with neighbors
    offsets:   (N+1,) LongTensor, CSR offsets
    tau:       int, cutoff timestamp
    k:         int, number of neighbors to sample
    replace:   bool, whether to sample with replacement

    Returns:
        nbrs:  (B, k) LongTensor (padded with -1)
        tvals: (B, k) LongTensor (padded with -1)
    """
    device = neighbors.device
    B = src_nodes.numel()

    # get neighbor index ranges for each src
    start = offsets[src_nodes]
    end   = offsets[src_nodes + 1]
    counts = end - start  # number of neighbors per node

    # build flat index of all candidate neighbors
    # arange = torch.arange(counts.sum(), device=device)
    # map each arange idx to its src node
    node_ids = torch.repeat_interleave(torch.arange(B, device=device), counts)

    # gather neighbor + time
    all_nbrs = neighbors[torch.cat([torch.arange(s, e, device=device) for s, e in zip(start, end)])]
    all_times = times[torch.cat([torch.arange(s, e, device=device) for s, e in zip(start, end)])]

    # filter by tau
    mask = all_times <= tau
    all_nbrs, all_times, node_ids = all_nbrs[mask], all_times[mask], node_ids[mask]

    # now group back by src node
    # build (B, k) outputs
    nbrs_out = torch.full((B, k), -1, dtype=torch.long, device=device)
    # tvals_out = torch.full((B, k), -1, dtype=torch.long, device=device)

    # sort neighbors by time (latest first for consistency)
    order = torch.argsort(all_times, descending=True, stable=True)
    all_nbrs, all_times, node_ids = all_nbrs[order], all_times[order], node_ids[order]

    # pack into output using segment fill
    idx_within = torch.zeros_like(node_ids)
    idx_within[1:] = (node_ids[1:] == node_ids[:-1]).cumsum(0)

    mask_k = idx_within < k
    all_nbrs, all_times, node_ids, idx_within = (
        all_nbrs[mask_k],
        all_times[mask_k],
        node_ids[mask_k],
        idx_within[mask_k],
    )

    nbrs_out[node_ids, idx_within] = all_nbrs
    # tvals_out[node_ids, idx_within] = all_times

    return nbrs_out #, tvals_out


def batch_indices_fast(src, dst, B, num_nodes=None):
    """
    Fast batching:
      - max batch size = B
      - no node repeats per batch
    Args:
        src, dst: LongTensor (E,)
        B: int
        num_nodes: int (optional, max node id+1 for mask)
    Returns:
        start, end: LongTensor batch boundaries
    """
    E = len(src)
    if num_nodes is None:
        num_nodes = max(src.max().item(), dst.max().item()) + 1

    used = torch.zeros(num_nodes, dtype=torch.bool)  # mask instead of set
    start = []
    end = []

    batch_start = 0
    count = 0

    for i in range(E):
        u, v = src[i].item(), dst[i].item()

        # close batch if full or conflict
        if count >= B or used[u] or used[v]:
            start.append(batch_start)
            end.append(i)
            batch_start = i
            used[:] = False
            count = 0

        used[u] = True
        used[v] = True
        count += 1

    # close last batch
    start.append(batch_start)
    end.append(E)

    return torch.tensor(start), torch.tensor(end)

#%% AGG

def scatter_mean_update(state_vec, idx, vals):
    """
    Perform gradient-preserving mean update on state_vec at indices idx.
    Handles repeated indices by averaging their updates.
    """
    # Sums across repeated indices
    sums = torch.zeros_like(state_vec).index_add(0, idx, vals)

    # Counts per index
    counts = torch.zeros(state_vec.size(0), device=vals.device).index_add(
        0, idx, torch.ones(idx.size(0), device=vals.device)
    )

    # Compute means (only where counts > 0)
    means = torch.where(
        counts[:, None] > 0, sums / counts[:, None], state_vec
    )

    # Update only active indices
    return torch.where(
        counts[:, None] > 0,
        means,
        state_vec
    )

def get_delta_t(src_event, neg_dst_event, t_event, neighbors, times, offsets, default=1e+11):
    """
    Compute delta t for negative event (src, neg_dst) at time t_event.

    Args:
        src_event     : LongTensor [B]  source nodes of events
        neg_dst_event : LongTensor [B]  negative destinations
        t_event       : LongTensor [B]  event times
        neighbors     : LongTensor [2E] neighbor list from build_neighbor_index_torch
        times         : LongTensor [2E] interaction times aligned with neighbors
        offsets       : LongTensor [N+1] index offsets for neighbors

    Returns:
        delta_t : FloatTensor [B]  time since last src–neg_dst interaction
    """
    device = neighbors.device
    B = src_event.size(0)
    delta_t = torch.full((B,), default, dtype=torch.float32, device=device)

    for i in range(B):
        u = src_event[i].item()
        v = neg_dst_event[i].item()
        t = t_event[i].item()

        start, end = offsets[u].item(), offsets[u+1].item()
        neighs = neighbors[start:end]
        times_u = times[start:end]

        # mask interactions with v before t
        mask = (neighs == v) & (times_u < t)
        if mask.any():
            last_t = times_u[mask].max()
            delta_t[i] = t - last_t

    return torch.log1p(delta_t)