# CTDG-SSM
This repository contains the reference implementation of the method described in the paper:

*CTDG-SSM: Continuous-time Dynamic Graph State-Space Models for Long Range Propagation*
Submitted to ICLR 2026.
---
## Overview
CTDG-SSM is a unified spatiotemporal state-space framework that integrates temporal memory compression with topology-aware propagation, enabling the long-range time and long-range space dependencies in CTDGs.

Note: To preserve anonymity for peer review, author names and affiliations are omitted.
---
## Requirements
- Python >= 3.10 
- torch  >= 2.5.1 
- torch-geometric>= 2.6.1 

Other dependencies listed in requirements.txt
Install dependencies via:
```bash
pip install -r requirements.txt
```

### Dataset Setup

The experimental data is provided in the `processed_data` directory. Each dataset is stored as a compressed `.7z` archive inside its respective subfolder.

To prepare the data:

1. Navigate to the corresponding subfolder under `processed_data/`.
2. Extract the `.7z` file in that subfolder.
3. After extraction, the folder should contain both the `.csv` and `.npy` files.

For example, for the **Enron** dataset, the structure should look like:
```
processed_data/
└── enron/
├── ml_enron.csv
└── ml_enron.npy
```
Make sure this structure is preserved for the code to run correctly.

### Running Experiments

This repository supports three main tasks:

- **Sequence Classification** → Run via [`ssm_seq_classification.ipynb`](ssm_seq_classification.ipynb)  
- **Node Classification** → Run via [`ssm_node_classification.py`](ssm_node_classification.py)  
- **Link Prediction** → Run via [`ssm_link_prediction.py`](ssm_link_prediction.py)  

### Code Structure

- **`ssm_memory.py`** — Defines the memory class with update mechanisms.  
- **`ssm_dataset.py`** — Implements the dataset class with sampling and batching utilities.  
- **`ssm_learn.py`** — Training code for link prediction.  
- **`ssm_nc_learn.py`** — Training code for node classification.  
- **`ssm_utils.py`** — Contains utility functions including neighbor sampling, Laplacian generation, and other graph operations.  
- **`utils/`** — Additional helper functions and utilities.  

Use the corresponding script/notebook depending on the task you want to run.  
