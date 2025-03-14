# Multi Modal Multi-(plexed/relational) Graph Neural Networks

Code for running experiments on Graph Classification (NIH-TB Dataset), and Node Classification (AIFB/ACM datasets on dgl) using the Multiplexed GNN and Maximal Correlation Multi-GNN
(Baselines lacking public code repositories were re-implemented)

## Directory Structure
```
MMMG/
| ── environment.yml:
| ── TB_data.pickle.zip # pre-processed TB portals data, needs to be unzipped
| ── Node Classification/ (Inductive setting on toy example datasets)
│  ├── ACM_Scripts/ # Baselines/Model for ACM classification task
|      └── Run_<model>.py #runs gnn predictors on ACM dataset
│  └── AIFB_Scripts/ # Baselines/Model for AIFB classification task
|      └── Run_<model>.py #runs gnn predictors on AIFB dataset
| ── Graph Classification/ (Transductive setting on real healthcare datasets)
│  └── utils/ # code for graph creation/processing/training/evaluation
|  └── TB_Scripts/ # code for experiment set up
     └── wrapper.py # command to run experiment with arguments
     └── Run_<model>.py/ # modules for baselines/multiplex/multiGNN
     └── Run_graph_prep.py # to create graph objects for various models that perform graph classification, needs to be run once before training for setup
```
## Data

NIH-TB portals pre-processed data can be unzipped into the parent directory as a pickle file, raw data can be downloaded from https://tbportals.niaid.nih.gov. 
The processed pickle file contains individual modality features and a mask indicating missing/unknown values. 

ACM/AIFB data are downloaded (once, then stored in cache) from dgl when you run the corresponding scripts under ```<project_dir>/Node_Classification/<dataset_name>_scripts/``` .

## Environment Setup

Clone the repository from git and instantiate an environment by running

```shell
conda env create -f environment.yml

```

## Running Experiments on Multimodal Datasets

Experiments for each model can be run by a call to wrapper.py under ```<project_dir>/Graph_Classification/TB_scripts/```, four arguments need to be specified

### Main Arguments

- `--model_type`: specify one of the fusion models: 'maxcorrmgnn','multiplex','rGCN', 'mGNN', 'multidimgcn', 'multibehGNN', 'GCN', 'latent_graph', 'nofusion', 'simple_fusion', 'metric_fusion', 'transformer'
- `--seed`: Seed value (0-9) for generating a cross validation split
- `--predictor_type` : required argument for maxcorrmgnn and rGCN baseline, specifies type of predictor, either GNN or MLP for HGR variants (MaxCorr MGNN vs HGR-ANN fusion baseline), and 'multiplex-like-graphs' or 'reduced-modality-hetero-graphs' for rGCN, will be ignored for other baselines
- `--loss_tradeoff`: loss tradeoff, set between 0-1 for maxcorrmgnn, setting this to 0.0 performs sequential training of HGR-projectors and Multi-GNN, also required as a parameter for Metric Fusion Network

### Other Arguments

- `--use_mlflow_tracking`: [0-No, 1-Yes] Optionally save experimental artefacts (AUC plots and values) on an MLflow log, results can be downloaded as a csv for statistical analysis
- `--process_graphs`: [0-No, 1-Yes] Create graph objects and store them locally, needs to be run once before running multiplex GNN and baselines from the corresponding paper, can be set to 0 after completion

  If you intend to use MLflow, please start a server in the terminal using the command below, and modify the url in line 48 in wrapper.py to reflect the same.

  ```shell
  mlflow server --host <hostname> --port <portnumber>
  ```
### Example Runs

First activate the environment:

```shell
conda activate MultiGraphNN
```

For Multiplexed GNN- load processed data, create graph objects as setup via modality and concept auto-encoders, then run graph classification using the multiplexed GNN predictor.

``` shell
python wrapper.py --model_type multiplex --seed 0 --predictor_type 'GNN' --loss_tradeoff 0 --use_mlflow_tracking 0 --process_graphs 1 
```

You will be prompted to provide the path to your local code directory before the script executes, and the directory path where you would like to store and retrieve assets from.

For MaxCorr MultiGNN - load processed data,, then run fully supervised node classification using the MaxCorrMGNN predictor

``` shell
python wrapper.py --model_type maxcorrmgnn --seed 0 --predictor_type 'GNN' --loss_tradeoff 0.01 --use_mlflow_tracking 0 --process_graphs 0 
```

## Citations

If you use this project in your research, please cite the following papers:
```
@inproceedings{d2023maxcorrmgnn,
  title={MaxCorrMGNN: A multi-graph neural network framework for generalized multimodal fusion of medical data for outcome prediction},
  author={D’Souza, Niharika S and Wang, Hongzhi and Giovannini, Andrea and Foncubierta-Rodriguez, Antonio and Beck, Kristen L and Boyko, Orest and Syeda-Mahmood, Tanveer},
  booktitle={Workshop on Machine Learning for Multimodal Healthcare Data},
  pages={141--154},
  year={2023},
  organization={Springer}
}

@article{d2024fusing,
  title={Fusing modalities by multiplexed graph neural networks for outcome prediction from medical data and beyond},
  author={D‘Souza, Niharika S and Wang, Hongzhi and Giovannini, Andrea and Foncubierta-Rodriguez, Antonio and Beck, Kristen L and Boyko, Orest and Syeda-Mahmood, Tanveer F},
  journal={Medical Image Analysis},
  volume={93},
  pages={103064},
  year={2024},
  publisher={Elsevier}
}

@inproceedings{d2022fusing,
  title={Fusing modalities by multiplexed graph neural networks for outcome prediction in tuberculosis},
  author={D’Souza, Niharika S and Wang, Hongzhi and Giovannini, Andrea and Foncubierta-Rodriguez, Antonio and Beck, Kristen L and Boyko, Orest and Syeda-Mahmood, Tanveer},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={287--297},
  year={2022},
  organization={Springer}
}
```


