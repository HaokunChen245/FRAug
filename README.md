# FRAug (Federated Representation Augmentation)

### Setup
See the requirements.txt for environment configuration.
```
pip install -r requirements.txt
```

### Dataset Preparation
Download the benchmark datasets using the provided script
```
python datasets/dataset_download.py --dataset="Digits" --data_dir=$PATH_TO_DATASETS$

python datasets/dataset_download.py --dataset="PACS" --data_dir=$PATH_TO_DATASETS$

python datasets/dataset_download.py --dataset="OfficeHome" --data_dir=$PATH_TO_DATASETS$
```

### Training
```
python main.py --dataset=$DATASET$ --dataset_dir=$PATH_TO_DATASETS$ --log_dir=$PATH_TO_LOG$
```

### Evaluation
```
python eval.py --dataset=$DATASET$ --model_dir=$PATH_TO_MODEL$ --dataset_dir=$PATH_TO_DATASETS$
```
