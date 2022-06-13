#!/bin/bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
unzip crawl-300d-2M-subword.zip

pip3 install upgrade --pip 

pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip3 install keras && numpy && sklearn && pandas && tensorflow && seaborn && fasttext && matplotlib && tqdm && xgboost
