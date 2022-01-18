# Object-ReID-Hierarchical-Neural-Networks

This repository contains the code for paper "Low-Power Multi-Camera Object Re-Identification using Hierarchical Neural Networks" (ISLPED 2021).

## Usage

### Dependencies
- [Python3.7](https://www.python.org/downloads/)
- [PyTorch(1.6.0)](http://pytorch.org)
- [torchvision(0.2.0)](http://pytorch.org)
- [Market 1501 dataset](http://www.liangzheng.org/Project/project_reid.html)
- [Market 1501 attributes](https://github.com/vana77/Market-1501_Attribute)
- [VRAI dataset and attributes](https://github.com/JiaoBL1234/VRAI-Dataset)

### Usage
#### Dataset Preprocessing
Use the following command to preprocess the person re-id dataset.

```bash
python create_market_dataset.py --path <root_path_of_dataset>
```


### Download the trained models
We provide our trained model. You may download it from [Google Drive](https://drive.google.com/drive/folders/1u6GkjuSoHPj7Xl7KTjeMIalbA4ZT-jHL?usp=sharing). You may download and move it to the `checkpoints` directory.


### Process and Evaluate on Entire Dataset
```bash
python .\tree_feature_extractor.py --data <root_path_of_dataset>
python .\tree_evaluate.py
```


### Demo on Single Query Image
```bash
python .\tree_feature_extractor.py --data <root_path_of_dataset>
python .\tree_demo.py --query <query_index>
```


## Citation
Please cite this paper if it helps your research:
```bibtex
@inproceedings{goel2021multi,
  title={Low-Power Multi-Camera Object Re-Identification using Hierarchical Neural Networks},
  author={Abhinav Goel, Caleb Tung, Xiao Hu, Haobo Wang, James C. Davis, George K. Thiruvathukal, Yung-Hsiang Lu},
  booktitle={Proceedings of the ACM/IEEE International Symposium on Low Power Electronics and Design},
  year={2021}
}
```

