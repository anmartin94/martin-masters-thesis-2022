"""Author: Anna Martin
Description: 
This program trains four BERT variants (bert-base and scibert, cased and uncased) on the given 
dataset. The following hyperparameters are applied in every combination: epochs (2, 3, 4), 
batch size (16, 32), and learning rate (2e-5, 3e-5, 5e-5). Each model and encoding combination is 
trained twenty times and the results are recorded. Ten of the iterations are trained using only 
sentence features. Ten of the iterations are trained using the sentence and positional features. The
validation F1 scores are averaged for each combination, and the best settings for each model are 
output as LaTeX code for a table in a txt file that can be copied and pasted into a LaTeX document
without further editing.
Usage: 
This program takes 5 arguments:
    1. The training dataset filepath
    2. The number of times to run each experiment
    3. Whether to use contextual features (1) or just text (0)
    4. Whether to use the cased version of the model (cased) or uncased (uncased)
    5. The name of the BERT model to use (scibert or bert)
    EXAMPLE: python3 bert-experiments.py datasets/reduced_train.csv 10 1 uncased scibert
Credit:
Huggingface's example code for text classification in pytorch was referenced when writing the
train_val_loop method:
https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py
as was Chris McCormick's blog:
https://mccormickml.com/2019/07/22/BERT-fine-tuning/
"""

import datetime
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
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
from transformers import get_linear_schedule_with_warmup

def concat_data(cols):
  """Takes an array of columns containing features and concatenates
  each feature into a single string. Returns an array of feature strings"""
  rows = []
  for i, (header, n1, n2, n3, n4, sentence) in enumerate(zip(cols[0], cols[1], cols[2], cols[3], cols[4], cols[5])):
    row = header + " " + str(n1) + " " +str(n2) + " " +str(n3) + " " +str(n4) + " " +sentence
    rows.append(row)
  return np.array(rows)


def preprocess_data(train_name, concat):
  """Gets the features from the train dataset. 
  Returns the training and validation features and labels."""
  train = pd.read_csv(train_name, sep='\t')
  target_val_size = round(len(train.index)*.1)
  print(target_val_size)
  paper_list = list(set(train["paper_id"]))
  val_paper_ids = []
  chosen = []
  random_choice = " "
  print(len(train.index))
  print(len(set(paper_list)))
  while len(val_paper_ids) < 10: #need to create 10 test-val splits
    new_set = []
    for i in range(21): #each val set contains 20 papers
        random_choice = random.choice(paper_list)
        while random_choice in chosen:
            random_choice = random.choice(paper_list)
        new_set.append(random_choice)
        chosen.append(random_choice)
    val_paper_ids.append(new_set)
  datasets = [] 
  val_df = None
  for val_set in val_paper_ids:
    for vs in val_set:
        if val_df is None:
            val_df = train[train["paper_id"] == vs]
        else:
            if len(val_df.index) + len(train[train["paper_id"]==vs])> target_val_size:
              lhs_diff = target_val_size - len(val_df.index)
              rhs_diff = len(val_df.index) + len(train[train["paper_id"]==vs]) - target_val_size
              if rhs_diff >= lhs_diff:
                break
            val_df = val_df.append(train[train["paper_id"] == vs])
    train_df = pd.concat([train, val_df]).drop_duplicates(keep=False)
    datasets.append([train_df.copy(), val_df.copy()])
    val_df = None

  processed = []
  for dataset in datasets:
      if concat: #If training on combined contextual and sentence features
        x_train = concat_data(
            [dataset[0]['headers'], dataset[0]['local_pos'], 
             dataset[0]['global_pos'], dataset[0]['local_pct'], 
             dataset[0]['global_pct'], dataset[0]['sentences']])
        x_val = concat_data(
            [dataset[1]['headers'], dataset[1]['local_pos'], 
             dataset[1]['global_pos'], dataset[1]['local_pct'], 
             dataset[1]['global_pct'], dataset[1]['sentences']])
      else: #If training on sentence feature only
        x_train = dataset[0]['sentences']
        x_val = dataset[1]['sentences']
      y_train = np.array(dataset[0]['labels'])
      y_val = np.array(dataset[1]['labels'])
      processed.append([x_train.copy(), x_val.copy(), y_train.copy(), y_val.copy()])
  return processed


def get_dataset_splits(datasets, tokenizer):
  """Tokenizes the training sequences using a BertTokenizer and
  creates an n_sets number of train-val splits. Returns a list of training and
  validation inputs, labels, and masks."""
  train_input_list, val_input_list, train_label_list, val_label_list = \
  [], [], [], []
  train_mask_list, val_mask_list = [], []
  for dataset in datasets:
      train_input_ids, train_attention_masks, val_input_ids, val_attention_masks = [], [], [], []
      for feature in dataset[0]:
        feature = tokenizer.encode(feature, add_special_tokens = True)
        train_input_ids.append(feature)
      train_input_ids = pad_sequences(train_input_ids, maxlen=256, dtype="long", 
                                value=0, truncating="post", padding="post")
      for sentence in train_input_ids:
          attention_mask = [int(token_id > 0) for token_id in sentence]
          train_attention_masks.append(attention_mask)
      train_label_list.append(dataset[2])
      for feature in dataset[1]:
        feature = tokenizer.encode(feature, add_special_tokens = True)
        val_input_ids.append(feature)
      val_input_ids = pad_sequences(val_input_ids, maxlen=256, dtype="long", 
                                value=0, truncating="post", padding="post")
      for sentence in val_input_ids:
          attention_mask = [int(token_id > 0) for token_id in sentence]
          val_attention_masks.append(attention_mask)
      val_label_list.append(dataset[3])
      
      train_input_list.append(train_input_ids)
      val_input_list.append(val_input_ids)
      train_mask_list.append(train_attention_masks)
      val_mask_list.append(val_attention_masks)
      
  """for i in range(1, n_sets+1):
    train_inputs, validation_inputs, train_labels, validation_labels = \
    train_test_split(input_ids, labels, random_state=i*11, test_size=0.1)
    train_masks, validation_masks, _, _ = \
    train_test_split(attention_masks, labels, random_state=i*11, test_size=0.1)
    train_input_list.append(train_inputs)
    val_input_list.append(validation_inputs)
    train_label_list.append(train_labels)
    val_label_list.append(validation_labels)
    train_mask_list.append(train_masks)
    val_mask_list.append(validation_masks)"""
  return train_input_list, val_input_list, train_label_list, val_label_list, \
  train_mask_list, val_mask_list

def train_val_loop(train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks, model_name, epochs, batch_size, lr, device):
  """Fine tunes the BERT model for sequence classification. Calculates the
  F1, precision, and recall scores on validation set. Returns scores after
  the last epoch."""
  train_inputs = torch.tensor(train_inputs)
  validation_inputs = torch.tensor(validation_inputs)

  train_labels = torch.tensor(train_labels)
  validation_labels = torch.tensor(validation_labels)

  train_masks = torch.tensor(train_masks)
  validation_masks = torch.tensor(validation_masks)

  train_data = TensorDataset(train_inputs, train_masks, train_labels)
  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(
      train_data, sampler=train_sampler, batch_size=batch_size)

  validation_data = TensorDataset(
      validation_inputs, validation_masks, validation_labels)
  validation_sampler = SequentialSampler(validation_data)
  validation_dataloader = DataLoader(
      validation_data, sampler=validation_sampler, batch_size=batch_size)
  model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, output_attentions=False, output_hidden_states=False)
  model.cuda()
  optimizer = AdamW(model.parameters(), lr = lr, eps=1e-8)
  total_steps = len(train_dataloader) * epochs
  scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps = 0,
                                              num_training_steps = total_steps)

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
          duration = str(
              datetime.timedelta(seconds=int(round((time.time()-start)))))
          print(
              '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.' \
              .format(step, len(train_dataloader), duration))
      b_input_ids = batch[0].to(device)
      b_input_mask = batch[1].to(device)
      b_labels = batch[2].to(device)
      model.zero_grad()        
      outputs = model(b_input_ids, 
                  token_type_ids=None, 
                  attention_mask=b_input_mask, 
                  labels=b_labels)
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
  
    print("Validating")
    start = time.time()
    model.eval()
    predictions, true_labels = [], []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in validation_dataloader:
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask, b_labels = batch
      with torch.no_grad():        
          outputs = model(b_input_ids, 
                          token_type_ids=None, 
                          attention_mask=b_input_mask)
      logits = outputs[0]
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()


      tmp_eval_accuracy = np.sum(
          np.argmax(logits, axis=1).flatten() == \
          label_ids.flatten())/len(label_ids.flatten())
      eval_accuracy += tmp_eval_accuracy
      nb_eval_steps += 1
      predictions.append(logits)
      true_labels.append(label_ids)
    duration = str(datetime.timedelta(seconds=int(round((time.time()-start)))))
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(duration))
    predictions = np.argmax(
        [item for sublist in predictions for item in sublist], axis=1).flatten()
    true_labels = [item for sublist in true_labels for item in sublist]
  torch.cuda.empty_cache()
  c_matrix = confusion_matrix(true_labels, predictions)
  c_matrix = matrix_to_str(c_matrix)
  print(f1_score(true_labels, predictions))
  return f1_score(
      true_labels, predictions), precision_score(
          true_labels, predictions), recall_score(true_labels, predictions), loss_values, c_matrix

def matrix_to_str(matrix):
    matrix_str = ""
    for row in matrix:
        row_str = ""
        for col in row:
            row_str += str(col)+" "
        matrix_str += row_str+"\n"
    return matrix_str

def get_best_results(results):
  """Takes a dictionary containing the results of every run for every 
  hyperparameter combination, calculates the mean F1 score, and stores only
  the results of the best hyperparameter combination for each model in a new
  dictionary that is returned"""
  best_results = {}
  for model in results:
    best_results[model] = {}
    best_epoch = None
    best_b_size = None
    best_lr = None
    best_f1 = 0
    best_f1std = None
    best_precision = None
    best_precisionstd = None
    best_recall = None
    best_recallstd = None
    for epochs in results[model]:
      for b_size in results[model][epochs]:
        for lr in results[model][epochs][b_size]:
          mean = np.mean(results[model][epochs][b_size][lr]['f1_scores'])
          if mean > best_f1:
            best_epoch = epochs
            best_b_size = b_size
            best_lr = lr
            best_f1 = mean
            best_f1std = np.std(
                results[model][epochs][b_size][lr]['f1_scores'])
            best_precision = np.mean(
                results[model][epochs][b_size][lr]['precision_scores'])
            best_precisionstd = np.std(
                results[model][epochs][b_size][lr]['precision_scores'])
            best_recall = np.mean(
                results[model][epochs][b_size][lr]['recall_scores'])
            best_recallstd = np.std(
                results[model][epochs][b_size][lr]['recall_scores'])
    best_results[model] = {
        'epochs': str(best_epoch), 'b_size': str(best_b_size), 'lr': str(best_lr), 
        'precision': {'mean': str(round(best_precision, 2)), 'std': str(round(best_precisionstd, 2))},
        'recall': {'mean': str(round(best_recall, 2)), 'std': str(round(best_recallstd, 2))},
        'f1': {'mean': str(round(best_f1, 2)), 'std': str(round(best_f1std, 2))}}
    return best_results

def write_all_results(results, caption, filename):
  latex_table = "\\begin{table}[]\n \\centering\n"+ \
  "\\begin{tabular}{|c|c|c|c|} \\hline\n"+ \
  " model & settings & metric & score"+ \
  "\\\\\\hline\\hline\n"
  for model in results:
    for epochs in results[model]:
      for b_size in results[model][epochs]:
        for lr in results[model][epochs][b_size]:
          latex_table += "\n\\multirow{3}{*}{"+model+"} & \\multirow{3}{*}{epochs="+str(epochs)+", batch size="+str(b_size)+", lr="+str(lr)+ \
          "} & Precision & "+ str(round(np.array(results[model][epochs][b_size][lr]["precision_scores"]).mean(), 2)) \
          +"\\pm"+ str(round(np.array(results[model][epochs][b_size][lr]["precision_scores"]).std(), 2)) +"\\\\\\hline"+ \
          "& & Recall & "+ str(round(np.array(results[model][epochs][b_size][lr]["recall_scores"]).mean(), 2)) \
          +"\\pm"+ str(round(np.array(results[model][epochs][b_size][lr]["recall_scores"]).std(), 2)) +"\\\\\\hline"+ \
                    "& & F1 & "+ str(round(np.array(results[model][epochs][b_size][lr]["f1_scores"]).mean(), 2)) \
          +"\\pm"+ str(round(np.array(results[model][epochs][b_size][lr]["f1_scores"]).std(), 2)) +"\\\\\\hline"
  latex_table += "\n\\end{tabular}\n \\caption{"+caption+"}\n \\label{tab:my_label}\n \\end{table}\n"
  with open(filename, 'w') as f:
    f.write(latex_table)
  f.close()



def write_latex_table(results, caption, filename):
  """Generates LaTeX code for the results and stores in a text file"""
  latex_table = "\\begin{table}[]\n \\centering\n"+ \
  "\\begin{tabular}{|c|c|c|c||c|c|} \\hline\n"+ \
  " model & epochs & batch size & learning rate & metric & score"+ \
  "\\\\\\hline\\hline\n"
  for model in results:
    latex_table += "\\multirow{3}{*}{"+model+"} & \\multirow{3}{*}{"+ \
    results[model]['epochs']+"} & \\multirow{3}{*}{"+results[model]['b_size']+ \
    "} & \\multirow{3}{*}{"+results[model]['lr']+"} & Precision & $"+ \
    results[model]['precision']['mean']+"\\pm"+results[model]['precision']['std']+ \
    "$\\\\\n & & & & Recall & $"+results[model]['recall']['mean']+ \
    "\\pm"+results[model]['recall']['std']+"$\\\\\n & & & & F1 & $"+ \
    results[model]['f1']['mean']+ \
    "\\pm"+results[model]['f1']['std']+"$\\\\\\hline\n"
    latex_table += "\\end{tabular}\n \\caption{"+caption+ \
  "}\n \\label{tab:my_label}\n \\end{table}\n"
  with open(filename, 'w') as f:
    f.write(latex_table)
  f.close()

def main():
  train_file = sys.argv[1]
  iterations = int(sys.argv[2])
  context = sys.argv[3] 
  cased = sys.argv[4]
  version = sys.argv[5]
  if torch.cuda.is_available():     
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  dataset_name = train_file[9:-10]

  if context == "1":
    context = True
    title = dataset_name+"-with-context"
  else:
    context = False
    title = dataset_name
  datasets = preprocess_data(train_file, context)
  results = {}
  if cased == "cased":
    if version == "scibert":
      models = ['allenai/scibert_scivocab_cased']
    else:
      models = ['bert-base-cased']
  else:
    if version == "scibert":
      models = ['allenai/scibert_scivocab_uncased']
    else:
      models = ['bert-base-uncased']
  learning_rates = [2e-5, 3e-5, 5e-5]
  batch_sizes = [16, 32]
  for model_name in models:
    results[model_name] = {}
    uncased=False
    if model_name[-7:] == 'uncased':
      uncased = True
      tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=uncased)
    train_input_list, val_input_list, train_label_list, val_label_list, train_mask_list, val_mask_list = get_dataset_splits(datasets, tokenizer)
    for epochs in range(2, 5):
      results[model_name][epochs] = {}
      for b_size in batch_sizes:
        results[model_name][epochs][b_size] = {}
        for lr in learning_rates:
          results[model_name][epochs][b_size][lr] = \
          {'f1_scores': [], 'precision_scores': [], 'recall_scores': []}
          results_key = model_name+"-"
          print("MODEL", model_name, "EPOCHS", epochs, "BATCH SIZE", b_size, \
                "LEARNING RATE", lr)
          losses = []
          for i in range(iterations):
            f1, precision, recall, loss_values, c_matrix = train_val_loop(
                train_input_list[i], val_input_list[i], train_label_list[i], \
                val_label_list[i], train_mask_list[i], val_mask_list[i], \
                model_name, epochs, b_size, lr, device)
            losses.append(loss_values)
            file_model_name = model_name.replace('/', '_')
            m_filename = "output/c_matrix-"+file_model_name+"-"+str(epochs)+"-"+str(b_size)+"-"+str(lr)+"-"+title+".txt"
            with open(m_filename, 'a') as f:
                f.write(str(i)+'\n')
                f.write(c_matrix)
                f.write('\n')
            results[model_name][epochs][b_size][lr]['f1_scores'].append(f1)
            results[model_name][epochs][b_size][lr]['precision_scores'].append(precision)
            results[model_name][epochs][b_size][lr]['recall_scores'].append(recall)
          if not os.path.isdir("output"):
            os.mkdir("output")
          str_to_print = ""
          png_name = "output/losses-"+title+"-"+file_model_name+"-"+str(epochs)+"-"+str(b_size)+"-"+str(lr)+"-"+datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S_')+".txt"
          for arr in losses:
            for i in arr:
              str_to_print += str(i)+" "
            str_to_print += "\n"
          with open(png_name, 'w') as f:
            f.write(str_to_print)
          f.close()
  best_results = get_best_results(results)
  caption = "BERT training results for "+title.replace('-', ' ')+"."
  allfilename = "output/all-results-"+title+"-"+datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S_')+".txt"
  filename = "output/best-results-"+title+"-"+datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S_')+".txt"
  write_all_results(results, "All-"+caption, allfilename)
  write_latex_table(best_results, caption, filename)

if __name__=="__main__":
  main()
