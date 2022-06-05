# martin-masters-thesis-2022

## NLPSharedTasks Corpus

### Data Statement
Following is the Data Statement for our corpus NLPSharedTasks (version 1).
#### A. Curation Rationale
Our corpus contains the full texts of 254 Shared Task Overview papers published in the ACL Anthology between the year 2000 and 2021. The criteria for inclusion are:
* The paper was written by the organizers of a Shared Task
* The paper provides a description of the Shared Task, including details on the dataset the task is performed over, the task to be implemented by participating systems, and an overview of participating systems
* The Shared Task described in the paper was hosted by some research workshop in the domain of computational linguistics or natural language processing (NLP)
These criteria ensure that the papers included in the corpus are likely to contain a Shared Task Description phrase. The ACL Anthology was chosen as the source because it provides a catalog that is easy to browse for qualifying candidates for inclusion. Furthermore, choosing a single anthology to draw from provided some consistency of paper style and organization. The starting year (2000) was chosen because the formatting of papers describing earlier initiatives was too dissimilar. 

#### B. Language Variety
The papers included in NLPSharedTasks are in English as used in scientific communication in linguistics, computer science, and natural language processing domains. 

#### C. Speaker Demographic
The demographics of the paper authors are unknown. The speakers are likely researchers and students of computational linguistics and natural language processing. 

#### D. Annotator Demographic
The annotation was performed by two English-speaking annotators well versed in a broad range of NLP topics. Annotator 1 is a graduate student in computer science with a B.S. in computer science, and annotator 2 is a post doctoral researcher in data science with a PhD in computer science. Both annotators had shared task experience. Neither annotator was compensated. 

#### E. Speech Situation
The papers included in NLPSharedTasks were written between 2000 and 2021 in research settings. The speech included in these papers is written and is assumed to be scripted and edited, as well as peer-reviewed. In the case of multiple authors, it is unknown whether interaction was either synchronous or asynchronous. The intended audience of the papers included in NLPSharedTasks is researchers and practitioners of computational linguistics and natural language processing. 

#### F. Text Characteristics
The genre of the texts included in NLPSharedTasks can be described as written scientific communication in computational linguistics domains and other fields. As such, scientific vocabulary is used throughout that is specific to these domains and the documents are structured in a formal way. Texts are structured with sections under headers including _Title_, _Abstract_, _Introduction_, _Related Work_, _Task Description_, _Results_, and _Conclusion_, among others.

#### G. Corpus Access
NLPSharedTasks is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

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
