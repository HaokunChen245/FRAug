# FRAug (Federated Representation Augmentation)

We proposed a representation augmentation method to improve the model performance in federated learning with feature shift across client local data. [ [PDF]](
https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_FRAug_Tackling_Federated_Learning_with_Non-IID_Features_via_Representation_Augmentation_ICCV_2023_paper.pdf)

![Method](/assets/FRAug.png)

---

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

---

### Citation
```bibtex
@InProceedings{Chen_2023_ICCV,
    author    = {Chen, Haokun and Frikha, Ahmed and Krompass, Denis and Gu, Jindong and Tresp, Volker},
    title     = {FRAug: Tackling Federated Learning with Non-IID Features via Representation Augmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {4849-4859}
}
```
