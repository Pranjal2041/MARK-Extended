# MARK-Extended

### Datasets

Simply run `python prepare_dataset.py` to download the datasets.

1. ImageData
2. ImageGridData
3. Market-CL-Data (Currently not downloaded and processed)
4. Cifar100

#### MD5 sums for Cifar100 h5py files
* c19b68529da058a7d338d4986d8b34c4  datasets/cifar100_test.h5
* dac25fc0818a98ee56356c88d1be252d  datasets/cifar100_train.h5
* d9fcd08a0a412d87105785b256d3aecf  datasets/cifar100_val.h5 

### Training

Currenyly, only main training with fixed config is partially implemented. Simply run `python train.py` for training the model. 