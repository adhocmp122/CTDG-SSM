
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
    edge_raw_features = edge_raw_features[1:]   # remove first invalid entry
    scale = 3600
    timestamps = train_data.node_interact_times / scale
    val_timestamps = val_data.node_interact_times / scale
    validation_time = val_timestamps.min()
    eval_time = val_timestamps.max()
    full_data.node_interact_times = full_data.node_interact_times / scale

    # Merge train + val
    train_data.node_interact_times = np.hstack((timestamps, val_timestamps))
    train_data.src_node_ids = np.hstack((train_data.src_node_ids, val_data.src_node_ids))
    train_data.dst_node_ids = np.hstack((train_data.dst_node_ids, val_data.dst_node_ids))

    # Encode time
    train_data.encoded_time = positional_encoding_mixer(train_data.node_interact_times, args.time_dim)
    full_data.encoded_time = positional_encoding_mixer(full_data.node_interact_times, args.time_dim)

    # Special handling for certain datasets
    if args.dataset in ['enron', 'uci']:
        edge_raw_features += 1
        edge_raw_features = edge_raw_features.mean(axis=-1)[:, None]

    # Add edge features
    edge_raw_features, tmx = add_edge_features(
        full_data.src_node_ids,
        full_data.dst_node_ids,
        full_data.node_interact_times,
        edge_raw_features,
        positional_encoding_mixer,
        args.time_dim
    )

    edge_feat_dim = edge_raw_features.shape[1]
    total_nodes = node_raw_features.shape[0]

    

    # Dataset wrapper
    TGD = TemporalEdgeDataset(
        train_data.src_node_ids,
        train_data.dst_node_ids,
        train_data.node_interact_times,
        edge_raw_features,
        train_data.encoded_time,
        T=args.batches,
        num_node=total_nodes,
        validation_time=validation_time,
        LastInteraction=tmx,
        context_window=args.context,
        mode=args.mode,
        device=args.device,
        task='NC',
        unique_batch=args.u_batch,
        labels=full_data.labels
    )
    # torch.autograd.set_detect_anomaly(True)

    # Dataset wrapper
    TGD_full = TemporalEdgeDataset(
        full_data.src_node_ids,
        full_data.dst_node_ids,
        full_data.node_interact_times,
        edge_raw_features,
        full_data.encoded_time,
        T=args.batches,
        num_node=total_nodes,
        validation_time=eval_time,
        LastInteraction=tmx,
        context_window=args.context,
        mode=args.mode,
        device=args.device,
        task='NC',
        unique_batch=args.u_batch,
        labels=full_data.labels
    )

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

        # Optimizer + training
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        val_auc_run = main(
            model=model,
            full_dataset=TGD_full,
            train_dataset=TGD,
            optimizer=optimizer,
            name=args.name,
            num_epochs=args.epochs,
            device=args.device
        )
        val_auc.append(val_auc_run)
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
    parser.add_argument("--time_dim", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embd_dim", type=int, default=8)

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
    