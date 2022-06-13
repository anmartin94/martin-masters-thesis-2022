# martin-masters-thesis-2022
This repository contains the corpus NLPSharedTasks we created and the experimental code we ran on NLPSharedTasks for the master's thesis _Annotating and Automatically Extracting Task Descriptions from Shared Task Overview Papers in Natural Language Processing Domains_. The corpus is provided in the directory `shared-task-corpus` and the experimental code is found in the directory `shared-task-classification-project`. For details on both projects, see `Martin-Thesis-2022.pdf`.

## NLPSharedTasks Corpus
NLPSharedTasks contains 254 Shared Task Overview papers in natural language processing domains published in the Association for Computational Linguistics Anthology between 2000 and 2021. For each paper, a single consecutive span of text is annotated as a Task Description Phrase (unless no qualifying spans are found in the paper). 

### Corpus Usage
Three forms of the corpus are provided:
* `pdf-dataset/` contains the original PDFs for the 254 un-annotated papers in the corpus;
* `xml-dataset/` contains the un-annotated papers in XML format as extracted by GROBID; and
* `annotated-txt-files/` contains the annotated text files extracted from the xml files. These files contain annotations of task description phrases. Each annotated sequence starts with the special token `<TASK>` and ends with the special token `</TASK>`.

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
pip install seaborn
pip install sklearn
pip install stanza
pip install tensorflow
pip install xgboost
```
It is necessary to install PyTorch. Note that the version of PyTorch will depend on whether a GPU is available and the version of CUDA. 
For a CPU run:
```bash
pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```
For CUDA version 11.3 run:
```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
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
python3 thesis-ml-experiments.py datasets/reduced-train.csv 10
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

### Run test.py
This file takes seven command line arguments:
* The file containing the training data
* Whether to use contextual data (1) or the simple dataset (0)
* Whether to use the cased (1) or uncased (0) version of BERT
* The specific BERT model
* The number of epochs
* The batch size
* The learning rate
It trains a BERT model using the provided hyperparameters and tests the trained model on three versions of the test set. The results for each version of the test set can be seen in the output files full.txt, auto_reduced.txt, and reduced.txt, with one line for each test sample. The scores are printed to the terminal.
```bash
python3 test.py reduced_train.csv 0 1 'allenai/scibert_scivocab_uncased' 4 32 5e-5
```

### About the Data
`shared-task-classification-project/` contains a subdirectory called `datasets/`. This directory contains preprocessed data for training and testing models on the NLPSharedTasks corpus. The provided training file is named `reduced_train.csv` because it is a smaller version from the original corpus. The training dataset was reduced by eliminating all paper sections that do not contain a task description. The dataset was reduced in this way to improve the balance between positive and negative samples. 

Three versions of the test dataset are provided. `full_test.csv` contains all sentences from each paper in the test set. `reduced_test.csv` was filtered in the same way as `reduced_train.csv`. `auto_reduced_test.csv` is a middle-ground dataset which was generated by using a model to predict whether a section is likely to contain a task description based on the section header. Sections classified as unlikely to contain a task description were removed from this dataset.

The .csv files contain 10 columns.
* Column 0: index relative to the file
* Column 1: sentence index relative to the entire corpus `id`
* Column 2: the document ID `paper_id`
* Column 3: the section header `headers`
* Column 4: the sentence index relative to the current section `local_pos`
* Column 5: the sentence index relative to the document `global_pos`
* Column 6: the quadrant of the current section in which the sentence is found `local_pct`
* Column 7: the quadrant of the document in which the sentence is found `global_pct`
* Column 8: the sentence `sentences`
* Column 9: the label `labels`

The data found in Columns 3-7 are used as training features in our experiments that use sentence context. This data is ignored in our experiments that learn only from the sentence data.

Data can be extracted into train and test csv files by running make_dataset.py. This program requires two arguments: the filepath to the folder containing the annotated text files, and the filepath to the folder where the output .csv files will be stored. For example:
```bash
python3 make-dataset.py shared-task-corpus/annotated-txt-files/ shared-task-classification-project/datasets/
```

