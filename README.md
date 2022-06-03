# martin-masters-thesis-2022

## Thesis Experiments

This repository contains the datasets and experiments created for Anna Martin's master's thesis.

### Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required python scripts.

```bash
pip install fasttext
pip install numpy
pip install pandas
pip install sklearn
pip install tensorflow
pip install xgboost
```
The script install.sh provided in the repository can be run to download the files required to use fasttext embeddings.
```bash
./install.sh
```
Or, the user may run
```bash 
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
unzip crawl-300d-2M-subword.zip
```
### Run thesis-ml-experiments.py 
This file takes two command line arguments. The first is the file path to the training file and the second is a number that dictates how many times each model and encoding combination is trained. 
```bash
python3 thesis-ml-experiments.py 10
```

### Run bert-experiments.py
This file takes five command line arguments:
* The training dataset filepath; 
* The number of times to run each experiment;
* Whether to use contextual features (1) or just text (0);
* Whether to use the cased version of the model (cased) or uncased (uncased); and
* The name of the BERT model to use (scibert or bert).
```bash
python3 bert-experiments.py datasets/reduced-train.csv 10 1 uncased scibert
```
