
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Changed working directory to:", os.getcwd())

import argparse
import json
import yaml
import torch

import numpy as np
from utils.DataLoader import get_node_classification_data
import scipy.sparse as sp
import torch.optim as optim
from ssm_memory import MemoryModel
from ssm_utlis import positional_encoding_mixer,add_edge_features
from ssm_dataset import TemporalEdgeDataset
from ssm_nc_learn import main


def load_config(config_path):
    """Load JSON or YAML config file."""
    if config_path.endswith(".json"):
        with open(config_path, "r") as f:
            return json.load(f)
    elif config_path.endswith((".yml", ".yaml")):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Config file must be .json, .yml or .yaml")

def enforce_types(args):
    """Ensure argparse + config values have the correct Python types."""
    args.lr = float(args.lr)
    args.wd = float(args.wd)
    args.val_ratio = float(args.val_ratio)
    args.test_ratio = float(args.test_ratio)

    args.time_dim = int(args.time_dim)
    args.hidden_dim = int(args.hidden_dim)
    args.embd_dim = int(args.embd_dim)
    args.batches = int(args.batches)
    args.context = int(args.context)
    args.epochs = int(args.epochs)
    args.runs = int(args.runs)

    args.reg = float(args.reg)
    args.delta = float(args.delta)
    args.u_batch = bool(args.u_batch)
    return args

def run_training(args):
    # -----------------------
    # Dataset Preparation
    # -----------------------
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data = \
        get_node_classification_data(dataset_name=args.dataset, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # Time normalization + encoding
    # edge_raw_features = edge_raw_features[1:]   # remove first invalid entry
    scale = 3600
    train_data.node_interact_times = train_data.node_interact_times / scale
    val_data.node_interact_times = val_data.node_interact_times / scale
    test_data.node_interact_times = test_data.node_interact_times / scale
    full_data.node_interact_times = full_data.node_interact_times / scale

    # Add edge features
    edge_raw_features, tmx = add_edge_features(
        full_data.src_node_ids,
        full_data.dst_node_ids,
        full_data.node_interact_times,
        edge_raw_features[1:],
        positional_encoding_mixer,
        args.time_dim
    )

    #add_dud
    edge_feat_dim = edge_raw_features.shape[1]
    zero_padd = np.zeros((1,edge_feat_dim))
    edge_raw_features = np.vstack((zero_padd,edge_raw_features))
    fin_tmx = np.zeros((len(tmx)+1,))
    fin_tmx[1:] = tmx
    tmx = fin_tmx 

    edge_feat_dim = edge_raw_features.shape[1]
    total_nodes = node_raw_features.shape[0]

    

    # Dataset wrapper
    TGD = TemporalEdgeDataset(
        train_data.src_node_ids,
        train_data.dst_node_ids,
        train_data.node_interact_times,
        edge_raw_features[train_data.edge_ids],
        biparted=False,
        T=args.batches,
        num_node=total_nodes,
        LastInteraction=tmx[train_data.edge_ids],
        context_window=args.context,
        mode='train',
        device=args.device,
        task='NC',
        unique_batch=args.u_batch,
        labels=train_data.labels
    )

        # Dataset wrapper

    # Merge train + val
    c_time = np.hstack((train_data.node_interact_times, val_data.node_interact_times))
    c_src = np.hstack((train_data.src_node_ids, val_data.src_node_ids))
    c_dst = np.hstack((train_data.dst_node_ids, val_data.dst_node_ids))


    TGD_val = TemporalEdgeDataset(
        val_data.src_node_ids,
        val_data.dst_node_ids,
        val_data.node_interact_times,
        edge_raw_features[val_data.edge_ids],
        biparted=False,
        T=args.batches,
        num_node=total_nodes,
        LastInteraction=tmx[val_data.edge_ids],
        context_window=args.context,
        mode='eval',
        device=args.device,
        task='NC',
        unique_batch=args.u_batch,
        labels=val_data.labels,
        c_dst= c_dst,
        c_time=c_time,
        c_src=c_src
    )

    c_dst = np.hstack((c_dst,test_data.dst_node_ids))
    c_src = np.hstack((c_src,test_data.src_node_ids))
    c_time = np.hstack((c_time,test_data.node_interact_times))

    TGD_test = TemporalEdgeDataset(
    test_data.src_node_ids,
    test_data.dst_node_ids,
    test_data.node_interact_times,
    edge_raw_features[test_data.edge_ids],
    biparted=False,
    T=args.batches,
    num_node=total_nodes,
    LastInteraction=tmx[test_data.edge_ids],
    context_window=args.context,
    mode='eval',
    device=args.device,
    task='NC',
    unique_batch=args.u_batch,
    labels=test_data.labels,
    c_dst=c_dst,
    c_src=c_src,
    c_time=c_time
    )

    # torch.autograd.set_detect_anomaly(True)

    print("Arguments used:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print(f'Training/ Testing model for {args.runs} runs')

    print(f'Hidden Dim: {args.hidden_dim}, Input Dim: {edge_feat_dim}, Context: {args.context}, Node Embedding dims: {args.embd_dim}')
    print(f'Training/ Testing model for {args.runs} runs')
    val_auc = []
    for run in range(args.runs):
        # -----------------------
        # Model
        # -----------------------
        model = MemoryModel(
            num_nodes=total_nodes,
            input_dim=edge_feat_dim,
            hidden_dim=args.hidden_dim,
            time_dim=args.time_dim,
            reg=args.reg,
            delta=args.delta,
            device=args.device,
            embd_dims=args.embd_dim,
            update_type='HiPPO'
        ).to(args.device)

        #  Count only trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Trainable parameters:", trainable_params)

        # Optimizer + training
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        test_auc = main(
            model=model,
            val_dataset=TGD_val,
            test_dataset=TGD_test,
            train_dataset=TGD,
            optimizer=optimizer,
            name=args.name,
            num_epochs=args.epochs,
            device=args.device
        )
        val_auc.append(test_auc)
    print(f'All Auc : {val_auc}')
    print(f'Test from runs Mean AUC: {np.mean(val_auc):.4f} Std AUC: {np.std(val_auc):.4f} ')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Node Classification Training")

    # Config file option
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file", default=None)

    # Dataset / files
    parser.add_argument("--dataset", type=str, default="reddit", help="Dataset name")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)

    # Model params
    parser.add_argument("--time_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--embd_dim", type=int, default=16)

    parser.add_argument("--context", type=int, default=10)
    parser.add_argument("--mode", type=str, default="FX")
    parser.add_argument("--reg", type=float, default=1e-3)
    parser.add_argument("--delta", type=float, default=1.0)

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)

    # Training
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--u_batch", type=bool, default=False)
    parser.add_argument("--runs", type=int, default=5)

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--name", type=str, default="Experiment_")

    args = parser.parse_args()

    # If config file is provided, load it and override defaults
    if args.config is not None:
        config_dict = load_config(args.config)
        for k, v in config_dict.items():
            setattr(args, k, v)
    args = enforce_types(args)
    run_training(args)
    