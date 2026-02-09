# CryoEM Map Enhancement

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

Official repository for the CryoEM Map Enhancement project. This repository contains code for **CryoPC** as well as reference methods. Built upon [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) and [CryoTEN](https://github.com/jianlin-cheng/cryoten).

## Table of Contents

- [Setup Repository](#setup-repository)
- [Project Structure](#project-structure)
- [Available Scripts](#available-scripts)
- [Running Experiments](#running-experiments)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Setup Repository

### Installation

Create a conda environment:

```bash
conda env create -f environment.yaml
```

Install required packages:

```bash
pip install -r requirements.txt
```

### Configuration

Create `configs/local/default.yaml`:

```yaml
storage_dir: /path/to/your_data_root
```

## Data setup
Download the maps from EMDB database. List of entries is stored in `data/testset.csv`. In order to run `eval_testset.py`, put this file inside of `your_data_root/data/`.

You can download model checkpoint from: `https://drive.google.com/drive/folders/1HeV5r51qydn-qjAUh50k2CyD3kyAsjRb?usp=sharing`.

You should place `cryopc.ckpt` inside of `your_data_root/models`.

## Project Structure

### Repository Structure

```
cryoem-map-enhancement/
├── configs/                  # Hydra configuration files
├── data/                     # Base CSV files for data download
├── logs/                     # Log storage
├── notebooks/                # Jupyter notebooks (not tracked in git)
├── scripts/                  # Utility scripts for data and evaluation
├── src/                      # Main codebase
│   ├── data/                 # Datamodules and data utilities
│   ├── models/               # Models and Lightning modules
│   ├── utils/                # Template utilities
│   └── evaluation_utils/     # Evaluation utilities
```

### Required Data Structure

```
your_data_root/
├── data/
│   └── dataset/              # CSV files from data/ directory
├── models/                   # Model checkpoints
└── outputs/                  # Output files
```

### Evaluation

**Single map:**

```bash
python3 -m src.eval evaluation="single_map_cryopc_main" input_map_path="your_input_path" output_map_path="your_save_path"
```

**Test set:**

```bash
python3 -m src.eval_testset evaluation="testset_cryopc_main"
```

You can also modify parameters like number of devices, checkpoint path, batch size and others using Hydra CLI API:

```bash
python3 -m src.eval_testset evaluation="testset_cryopc_main" ckpt_path="/my/ckpt/path.ckpt" trainer="gpu" trainer.devices=2
```

## Citation

```bibtex
@article {Karanowski2026.01.17.700079,
	author = {Karanowski, Konrad and Chojecki, Mi{\l}osz and Rzepiela, Andrzej and Grzesiuk, Mateusz and Kuczba{\'n}ski, Rados{\l}aw and Frey, Lukas and Zi{\k e}ba, Maciej},
	title = {Point-Cloud Enhancement and Structural Interpretation Framework for Cryo-EM Maps},
	elocation-id = {2026.01.17.700079},
	year = {2026},
	doi = {10.64898/2026.01.17.700079},
	URL = {https://www.biorxiv.org/content/early/2026/01/21/2026.01.17.700079},
	eprint = {https://www.biorxiv.org/content/early/2026/01/21/2026.01.17.700079.full.pdf},
	journal = {bioRxiv}
}

```
---

## License

This project is licensed under the MIT License.

## Acknowledgments

This repository builds upon:
- [Lightning-Hydra Template](https://github.com/ashleve/lightning-hydra-template)
- [CryoTEN](https://github.com/jianlin-cheng/cryoten)

Additional resources used:
- [PointNet](https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py)
- [Cross-Attention UNet](https://github.com/chenwei-zhang/CryoSAMU/tree/main)
