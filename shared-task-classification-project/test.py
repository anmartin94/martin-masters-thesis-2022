"""
Author: Anna Martin
Description: 
This program trains a BERT model using the given hyperparameters and tests the model on three
versions of the test set from the annotated NLPSharedTasks corpus. It outputs one results file
for each version of the test set. These files contain each test sample, one per line. Following the
sentence is the true label and the predicted label, seperated by a tab.
Usage: 
This program takes 7 command line arguments:
1.) The file containing the training data
2.) Whether to use contextual data (1) or the simple dataset (0)
3.) Whether to use the cased (1) or uncased (0) version of BERT
4.) The specific BERT model
5.) The number of epochs
6.) The batch size
7.) The learning rate
EXAMPLE: python3 test.py reduced_train.csv 0 1 'allenai/scibert_scivocab_uncased' 4 32 5e-5
"""
import datetime
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pandas as pd
import random
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import sys
import tensorflow as tf
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data import TensorDataset
from transformers import AdamW
from transformers import BertConfig
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import DebertaTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel


def concat_data(cols):
    """Takes an array of columns containing features and concatenates
    each feature into a single string. Returns an array of feature strings"""
    rows = []
    for i in range(len(cols[0])):
        row = ""
        for col in cols:
          row += str(col[i])+' '
        rows.append(row)
    return np.array(rows)

def preprocess_data(train_name, test_name, concat):
    """Extracts training features and class labels from the training and 
    validation data and concatenates training and validations sets"""
    train = pd.read_csv(train_name, sep='\t')
    test = pd.read_csv(test_name, sep='\t')
    if concat: #If training on combined contextual and sentence features
        x_train = concat_data([train['headers'], train['local_pos'], train['global_pos'], train['local_pct'], train['global_pct'], train['sentences']])
        x_test = concat_data([test['headers'], test['local_pos'], test['global_pos'], test['local_pct'], test['global_pct'], test['sentences']])
    else: #If training on sentence feature only
        x_train = np.array(train['sentences'])
        x_test = np.array(test['sentences'])
        y_train = np.array(train['labels'])
        y_test = np.array(test['labels'])
    return x_train, y_train, x_test, y_test
  
def get_inputs_masks(train_x, train_y, test_x, test_y, tokenizer):
    """Tokenizes the training sequences using a BertTokenizer and
    creates an n_sets number of train-val splits. Returns a list of training and
    validation inputs, labels, and masks."""
    train_input_ids, train_attention_masks, test_input_ids, test_attention_masks = [], [], [], []
    for feature in train_x:
        feature = tokenizer.encode(feature, add_special_tokens = True)
        train_input_ids.append(feature)
    train_input_ids = pad_sequences(train_input_ids, maxlen=256, dtype="long", value=0, truncating="post", padding="post")
    for sentence in train_input_ids:
        train_attention_mask = [int(token_id > 0) for token_id in sentence]
        train_attention_masks.append(train_attention_mask)
    for feature in test_x:
        feature = tokenizer.encode(feature, add_special_tokens = True)
        test_input_ids.append(feature)
    test_input_ids = pad_sequences(test_input_ids, maxlen=256, dtype="long", 
                            value=0, truncating="post", padding="post")
    for sentence in test_input_ids:
        test_attention_mask = [int(token_id > 0) for token_id in sentence]
        test_attention_masks.append(test_attention_mask)
    train_labels = train_y
    test_labels = test_y
    return train_input_ids, test_input_ids, train_attention_masks, test_attention_masks, train_labels, test_labels
  
def train_loop(train_inputs, train_labels, train_masks, model_name, epochs, batch_size, lr, device):
    """Fine tunes the BERT model for sequence classification. Calculates the
    F1, precision, and recall scores on validation set. Returns scores after
    the last epoch."""
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, output_attentions=False, output_hidden_states=False)
    model.cuda()
    optimizer = AdamW(model.parameters(), lr = lr, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    loss_values = []

    for epoch_i in range(0, epochs):
        print('Epoch', epoch_i + 1, '/', epochs)
        print('Training')
        start = time.time()
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 20 == 0 and step != 0:
                duration = str(datetime.timedelta(seconds=int(round((time.time()-start)))))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), duration))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()        
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)            
        loss_values.append(avg_train_loss)
        duration = str(datetime.timedelta(seconds=int(round((time.time()-start)))))
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(duration))
    return model
  

def test(test_inputs, test_labels, test_masks, batch_size, device, model, test_x, output_name):
    print(len(test_inputs), len(test_labels), len(test_masks))
    test_inputs = torch.tensor(test_inputs)
    test_labels = torch.tensor(test_labels)
    test_masks = torch.tensor(test_masks)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    print("Testing")
    start = time.time()
    model.eval()
    predictions, true_labels = [], []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():        
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()


        tmp_eval_accuracy = np.sum(np.argmax(logits, axis=1).flatten() == label_ids.flatten())/len(label_ids.flatten())
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
        predictions.append(logits)
        true_labels.append(label_ids)
    duration = str(datetime.timedelta(seconds=int(round((time.time()-start)))))
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(duration))
    predictions = np.argmax([item for sublist in predictions for item in sublist], axis=1).flatten()
    true_labels = [item for sublist in true_labels for item in sublist]

    with open(output_name, 'w') as f:
        for x, y, z in zip(test_x, true_labels, predictions):
            f.write(x+"\t"+str(y)+"\t"+str(z)+"\n")
    f.close()
    print(f1_score(true_labels, predictions))
    return f1_score(true_labels, predictions), precision_score(true_labels, predictions), recall_score(true_labels, predictions)
  
def run(train_file, test_file, context, cased, model_name, epochs, batch_size, learning_rate, from_saved=False, model=None):
    if torch.cuda.is_available():     
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset_name = train_file[9:-10]

    if context == 1:
        context = True
        title = dataset_name+"-with-context"
    else:
        context = False
        title = dataset_name
    uncased = False
    if cased == 0:
        uncased = True
    
    train_x, train_y, test_x, test_y = preprocess_data(train_file, test_file, context)

    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=uncased)
    train_input_ids, test_input_ids, train_attention_masks, test_attention_masks, train_labels, test_labels = get_inputs_masks(train_x, train_y, test_x, test_y, tokenizer)
    if not from_saved:
        model = train_loop(train_input_ids, train_labels, train_attention_masks, model_name, epochs, batch_size, learning_rate, device)
        out_file = test_file[:-9]+".txt"
        f1, precision, recall = test(test_input_ids, test_labels, test_attention_masks, batch_size, device, model, test_x, out_file)
        print("Precision:", precision, "Recall:", recall, "F1:", f1)
    return model
    
def main():
    train_filepath = sys.argv[1]
    context = int(sys.argv[2])
    cased = int(sys.argv[3])
    bert = sys.argv[4]
    epochs = int(sys.argv[5])
    batch_size = int(sys.argv[6])
    lr = float(sys.argv[7])
    model = run(train_filepath, "reduced_test.csv", context, cased, bert, epochs, batch_size, lr)
    run(train_filepath, "auto_reduced_test.csv", context, cased, bert, epochs, batch_size, lr, from_saved=True, model=model)
    run(train_filepath, "full_test.csv", context, cased, bert, epochs, batch_size, lr, from_saved=True, model=model)
    
if __name__=="__main__":
  main()
    
