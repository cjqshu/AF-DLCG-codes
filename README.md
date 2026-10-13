# ConfounderDCI

This is the official implementation of the paper **"[AF-DLCG]()"**

## Project Directory Structure

```sh
AF-DLCG-codes/
│  README.md
├─casual-prediction-task
│  │  causal_discovery.py
│  │  data_loader.py
│  │  main_CPT.py
│  │  requirements.txt
│  │  utils.py
│  ├─dataset
│  └─models
│          bnncc_regression.py
└─causal-graphs-fusion-task
    │  requirements.txt
    ├─causa_graphs_fusion
    │  │  fusion_CD_algs_LLMs.ipynb
    │  ├─CD_algs
    │  └─CD_LLMs
    ├─data
    └─utils
```

## 🚀 How To Run


### Causal Graphs Fusion Task

- Installation: environment and dependences.
```sh
## Set up a new conda environment with Python 3.9
conda create -n CGfusion python=3.9
conda activate CGfusion

## Install python libraries or dependences.
# jupyter==1.1.1
# networkx==3.2.1
# pandas==2.3.3
# seaborn==0.13.2
# causal-learn==0.1.4.3
# gcastle==1.0.4  
# pgmpy==1.0.0    
# dowhy==0.14
# econml==0.16.0
# causalml==0.15.5
# pygraphviz==1.10
# pdf2image==1.17.0  
# rpy2==3.6.4

pip install -r requirements.txt
```


- Evaluation: quick start.

```sh
run fusion_CD_algs_LLMs.ipynb  # Causal Graphs Fusion
```


### Casual Prediction Task

- Installation: environment and dependences.
```sh
## Set up a new conda environment with Python 3.9
conda create -n bnncc python=3.9
conda activate bnncc

## Install python libraries or dependences.
# gcastle==1.0.3
# torch==2.1.0
# configargparse==1.7
# graphviz==0.20.3
# jupyter==1.1.1
# lazypredict==0.2.13
# tensorflow-gpu==2.10.0
# angles==2.0
# numpy==1.24.3
# ICPy==0.0.6
# causalicp==0.1.1
# sampler==0.2.14
# seaborn==0.13.2
# scikit-posthocs==0.11.4
# autorank==1.3.0
# SciencePlots==2.2.0
# ydata-profiling==4.17.0

pip install -r requirements.txt
```


- Evaluation: quick start.

```sh
python main_CPT.py  # Casual Prediction
```



