import os
import pandas as pd

df = pd.read_csv('reduced_train.csv', sep='\t')
current_paper_id = ""
tp, fp, tn, fn = 0, 0, 0, 0
for idx, row in df.iterrows():
    if current_paper_id == row['paper_id']:
        if row['labels'] == 1:
            fn += 1
        else:
            tn += 1
        continue
    current_paper_id = row['paper_id']
    if row['labels'] == 1:
        tp += 1
    else:
        fp += 1
print(tp, fp, tn, fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2*precision*recall/(precision+recall)
print(precision, recall, f1)
