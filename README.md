# pytorch-lightning(pl) with mlflow server

## main contribution using pl with mlflow

- managing and tracking experiment hyperparams
- comparing results across runs
- checkpointing, resuming training
- usable in both training clusters and notebooks/colab type settings

# Prerequisites
Main Prerequisites versions
````sh
pip install torch==1.11.0
pip install pytorch-lightning==1.6.0
pip install timm==0.5.4
pip install albumentations==1.1.0
pip install mlflow==1.24.0
pip install jsonargparse==4.5.0
pip install torchmetrics==0.7.3
pip install pandas==1.4.2
````

using conda(recommend but not tested yet)
```sh
# Make sure a name, prefix in .yml file
conda env create --file environment.yml
```

using pip(not tested)
```sh
pip install -r requirements.txt
```

## MLFlow
Command to run mlflow server with sql DB
```sh
# run mlflow server(localhost or remote url)
mlflow server --backend-store-uri [SQLITE] --default-artifact-root [DIRECTORY] -h [IP] -p [PORT]
```

Example
```sh
# run mlflow server example
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts -h 0.0.0.0 -p 5000
```

## TRAINS
```sh
python train.py --config config.yaml
```

## Make yaml file using LightningCLI in pytorch-lightning(>=1.6.0)
```sh
python train.py --print_config > args.yaml
```
