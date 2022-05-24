"""
Author: Anna Martin
Description: 
This program trains a set of machine learning models on the given dataset using a set of encodings.
Each model and encoding combination is trained twenty times and the results are recorded. Ten of the
iterations are trained using only sentence features. Ten of the iterations are trained using the
sentence and positional features.
Usage: 
The names of the train and validation files must be passed as command line arguments,
as well as the number of iterations per model per encoding type.
The program prints model performance summaries and training progress and outputs the average
precision, recall, and f1 scores for each model for each encoding of the dataset to two .txt files,
one for the model trained only on sentence data and the other for the model trained with positional
data.
Credit:
The selection of models and encodings is based off of the pipeline description found here:
https://towardsdatascience.com/binary-and-multiclass-text-classification-auto-detection-in-a-model-test-pipeline-938158854943
by Christophe Pere, whose scripts and notebook can be found here:
https://github.com/Christophe-pere/Text-classification
"""

from datetime import datetime
import fasttext
import fasttext.util
from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re, random
from sklearn import decomposition
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import explained_variance_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
import sys
import tensorflow as tf
from tensorflow import keras
import time
from tqdm import tqdm
import xgboost as xgb
from xgboost import XGBClassifier
import copy
import seaborn as sns
import matplotlib.pyplot as plt

def concat_data(cols):
  """Takes an array of columns containing features and concatenates
  each feature into a single string. Returns an array of feature strings"""
  rows = []
  for i, (header, n1, n2, n3, n4, sentence) in enumerate(zip(cols[0], cols[1], cols[2], cols[3], cols[4], cols[5])):
    row = header + " " + str(n1) + " " +str(n2) + " " +str(n3) + " " +str(n4) + " " +sentence
    rows.append(row)
  return np.array(rows)

#def extract_datasets(train_name, val_name, concat):
def extract_datasets(train_name):
  """Gets the features from the train and validation sets. 
  Returns the training and validation features and labels.
  (NOTE: this program does not use these splits.)"""
  train = pd.read_csv(train_name, sep='\t')
  target_val_size = round(len(train.index)*.1)
  paper_list = list(set(train["paper_id"]))
  val_paper_ids = []
  chosen = []
  random_choice = " "
  while len(val_paper_ids) < 10:
    new_set = []
    for i in range(21):
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
            if len(val_df.index)+len(train[train["paper_id"]==vs])>target_val_size:
                lhs_diff = target_val_size - len(val_df.index)
                rhs_diff = len(val_df.index) + len(train[train["paper_id"]==vs])-target_val_size
                if rhs_diff >= lhs_diff:
                    break
            val_df = val_df.append(train[train["paper_id"] == vs])
    train_df = pd.concat([train, val_df]).drop_duplicates(keep=False)
    datasets.append([train_df, val_df])
  #val = pd.read_csv(val_name, sep='\t')
  processed_with_context = []
  processed_no_context = []
  for dataset in datasets:
      #if concat: #If training on combined contextual and sentence features
    x_train_context = concat_data(
        [dataset[0]['headers'], dataset[0]['local_pos'], 
         dataset[0]['global_pos'], dataset[0]['local_pct'], 
         dataset[0]['global_pct'], dataset[0]['sentences']])
    x_val_context = concat_data(
        [dataset[1]['headers'], dataset[1]['local_pos'], 
         dataset[1]['global_pos'], dataset[1]['local_pct'], 
         dataset[1]['global_pct'], dataset[1]['sentences']])
        #x_val = concat_data(
            #[val['headers'], val['local_pos'],
             #val['global_pos'], val['local_pct'], 
             #val['global_pct'], val['sentences']])
      #else: #If training on sentence feature only
    x_train_no_context = dataset[0]['sentences']
    x_val_no_context = dataset[1]['sentences']
        #x_val = val['sentences']
    y_train = np.array(dataset[0]['labels'])
    y_val = np.array(dataset[1]['labels'])
      #y_val = np.array(val['labels'])
      #return x_train, y_train, x_val, y_val
    processed_with_context.append([x_train_context.copy(), x_val_context.copy(), y_train.copy(), y_val.copy()])
    processed_no_context.append([x_train_no_context.copy(), x_val_no_context.copy(), y_train.copy(), y_val.copy()])
  return processed_with_context, processed_no_context

#def get_df(x_train, y_train, x_val, y_val):
  #def get_df(x, y):
def get_df(train_file):
  original_df = pd.read_csv(train_file, sep='\t')
  """Stores the features and labels in a dataframe, 
  combining the training and validation data (this program
  creates its own splits. Returns the dataframe.)"""
  df = pd.DataFrame(data=[original_df["sentences"], original_df["labels"]], 
                    index=["text", "label"]).T
  return df

def get_splits(n_folds, df):
  """Creates an n_folds number of splits and stores each train/val
  split in a list called datasets. Datasets is returned"""
  datasets = []
  for i in range(1,n_folds+1):
    train_x, valid_x, y_train, y_valid = \
    model_selection.train_test_split(
        df['text'], df['label'], random_state=11*i, stratify=df['label'], test_size=0.1)
    datasets.append([train_x, valid_x, y_train, y_valid])
  return datasets

def init_encodings(df):
  """This method does all the work of initializing and
  fitting each encoding used in these experiments (one-hot, tf-idf,
  tf-idf character-level ngrams, tf-idf word-level ngrams, and fasttext
  embeddings). Should probably be broken up but it works. Returns the label
  encoder, the count vectors (one-hot encoding), all three tf-idf vectors, 
  fasttext embeddings, and the tokenizer."""
  pretrained = fasttext.FastText.load_model('crawl-300d-2M-subword.bin')

  encoder = preprocessing.LabelEncoder()

  count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
  count_vect.fit(df['text'])


  tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
  tfidf_vect.fit(df['text'])


  tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
  tfidf_vect_ngram.fit(df['text'])


  tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char',  ngram_range=(2,3), max_features=5000) 
  tfidf_vect_ngram_chars.fit(df['text'])


  token = Tokenizer()
  token.fit_on_texts(df['text'])
  word_index = token.word_index



  embedding_matrix = np.zeros((len(word_index) + 1, 300))
  for word, i in tqdm(word_index.items()):
      embedding_vector = pretrained.get_word_vector(word) 
      if embedding_vector is not None:
          embedding_matrix[i] = embedding_vector
  embedded = keras.layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)

  return encoder, count_vect, tfidf_vect, tfidf_vect_ngram, tfidf_vect_ngram_chars, embedded, token

def init_models(embedded):
  """Initializes each machine learning model used in these experiments (
    naive bayes, logistic regression, support vector machine, k-nearest
    neighbors, random forest, stochastic gradient descent, gradient boosting
    xgboost, a basic shallow neural network, a basic deep neural network, 
    a recurrent neural network, a convolutional neural net, and 
    an lstm and bilstm). Each model is stored in a dictionary and returned."""
  nn_model = keras.Sequential(
  [embedded, keras.layers.GlobalAveragePooling1D(), keras.layers.Dense(1, activation='sigmoid')])
  dnn_model = keras.Sequential(
  [embedded, keras.layers.GlobalAveragePooling1D(), keras.layers.Dense(16, activation='relu'), keras.layers.Dense(1, activation='sigmoid')])
  rnn_model = keras.Sequential(
  [embedded, keras.layers.SimpleRNN(32, return_sequences=True), 
                                keras.layers.SimpleRNN(32, return_sequences=True), 
                                keras.layers.SimpleRNN(32, return_sequences=True), 
                                keras.layers.SimpleRNN(32), keras.layers.Dense(1, activation='sigmoid')])
  cnn_model = keras.Sequential([
      embedded,
      keras.layers.Conv1D(100, 5, activation='relu'),
      keras.layers.Dropout(0.2),
      keras.layers.MaxPooling1D(pool_size=4),
      keras.layers.Conv1D(64, 5, activation='relu'),
      keras.layers.Dropout(0.2),
      keras.layers.MaxPooling1D(pool_size=4),
      keras.layers.Conv1D(32, 5, activation='relu'),
      keras.layers.Dropout(0.2),
      keras.layers.GlobalMaxPooling1D(),
      keras.layers.Dense(1, activation='sigmoid')])
  lstm_model = keras.Sequential([
      embedded,
      keras.layers.LSTM(32),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(1, activation='sigmoid')])
  bilstm_model = keras.Sequential([
      embedded,
      keras.layers.Bidirectional(keras.layers.LSTM(32)),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(1, activation='sigmoid')])

  models = {'naive_bayes': naive_bayes.ComplementNB(), 
            'logistic_regression': linear_model.LogisticRegression(max_iter=1000,  random_state=42), 
            'svm': svm.LinearSVC(), 
            'knn': neighbors.KNeighborsClassifier(n_neighbors=20, weights='distance', n_jobs=-1), 
            'random_forest': ensemble.RandomForestClassifier(), 
            'sgd': SGDClassifier(
            loss='modified_huber', max_iter=1000, tol=1e-3,   n_iter_no_change=10, early_stopping=True, n_jobs=-1 ), 
            'gradient_boost':ensemble.GradientBoostingClassifier(
            n_estimators=1000, validation_fraction=0.2, n_iter_no_change=5, tol=0.01, random_state=0, verbose=0),
            'xgb': XGBClassifier(n_estimators=1000, subsample=0.8),
            'nn': nn_model,
            'dnn': dnn_model, 
            'rnn': rnn_model, 
            'cnn': cnn_model,
            'lstm': lstm_model,
            'bilstm': bilstm_model
  }
  return models

def train_loop(models, datasets, encoder, count_vect, tfidf_vect, tfidf_vect_ngram, tfidf_vect_ngram_chars, embedded, token, context, n_iter):
  """This method takes a dictionary containing the models, a list containing
  each train-val split, each encoding, and the tokenizer. It iteratively
  trains each model on ten versions of the dataset for each encoding that is
  appropriate for the model. It stores the validation results (precision,
  recall, and f1 scores) as a list for each model / encoding combination in a 
  dictionary and returns the results."""
  labels = ["non-task description", "task description"]
  #the non-neural and neural models have different needs. These lists are
  #used to lookup each model to determine what kind of encodings it should
  #be trained on
  non_neural = ['naive_bayes', 'logistic_regression', 'svm', 'knn', 'random_forest']
  deep = ['nn', 'dnn', 'rnn', 'cnn', 'lstm', 'bilstm']
  ml_results = {}
  for model in models:
    ml_results[model] = {
        'one_hot': {'precision': [], 'recall': [], 'f1': []},
         'tfidf': {'precision': [], 'recall': [], 'f1': []},
         'tfidf_word_ngrams': {'precision': [], 'recall': [], 'f1': []},
         'tfidf_char_ngrams': {'precision': [], 'recall': [], 'f1': []},
         'embeddings': {'precision': [], 'recall': [], 'f1': []},
    }
    i = 0 #tracks the iterations                
    for dataset in datasets:
      i += 1 
      if i > n_iter:
        break

      # get the train/val splits from dataset
      train_x, valid_x = dataset[0], dataset[1]
      y_train, y_valid = dataset[2], dataset[3]

      # transform the data for each encoding
      train_y = encoder.fit_transform(y_train) 
      valid_y = encoder.fit_transform(y_valid) 
      xtrain_count = count_vect.transform(train_x) 
      xvalid_count = count_vect.transform(valid_x) 
      xtrain_tfidf =  tfidf_vect.transform(train_x) 
      xvalid_tfidf =  tfidf_vect.transform(valid_x) 
      xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x) 
      xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x) 
      xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
      xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 
      train_seq_x = sequence.pad_sequences(
          token.texts_to_sequences(train_x), maxlen=300) 
      valid_seq_x = sequence.pad_sequences(
          token.texts_to_sequences(valid_x), maxlen=300) 


      encodings = { 
          'one_hot': [xtrain_count, xvalid_count],
          'tfidf': [xtrain_tfidf, xvalid_tfidf],
          'tfidf_word_ngrams': [xtrain_tfidf_ngram, xvalid_tfidf_ngram],
          'tfidf_char_ngrams': [xtrain_tfidf_ngram_chars, xvalid_tfidf_ngram_chars],
          'embeddings': [train_seq_x, valid_seq_x]
      }
      
      for encoding in encodings:
        print("Training", model, "model with", encoding, 
              "encoding:", i, "/", len(datasets))
        if model in non_neural and encoding == 'embeddings':
          continue
        elif model in deep and encoding != 'embeddings':
          continue
        start = time.time()
        models_copy = copy.deepcopy(models)
        model_to_fit = models_copy[model]
        if model in deep:
          if model == 'cnn':
            optimizer = tf.keras.optimizers.RMSprop(lr=1e-4)
            epochs = 50
          elif model == 'lstm' or model == 'bilstm':
            optimizer = 'adam'
            epochs = 50
          else:
            optimizer = 'adam'
            epochs = 1000
          model_to_fit.compile(optimizer=optimizer, 
                                loss=tf.losses.BinaryCrossentropy(from_logits=True), 
                                metrics=['accuracy'])
          fit = model_to_fit.fit(encodings[encoding][0], train_y, 
                                  epochs=epochs, 
                                  callbacks=[tf.keras.callbacks.EarlyStopping(
                                      monitor='val_loss', mode='auto', patience=3)], 
                                  validation_split=.2, verbose=True)
          results = model_to_fit.evaluate(encodings[encoding][1], valid_y)
        else:
          model_to_fit.fit(encodings[encoding][0], train_y)
        end = time.time() - start
        predictions = model_to_fit.predict(encodings[encoding][1])
        if model not in deep:
          score = model_to_fit.score(encodings[encoding][1], valid_y)
          print(f"Score : {round(100*score,2)} %" )
        print("Execution time : %.3f s" %(end))
        print("\nClassification Report\n")
        if model in deep:
          predictions = (predictions>.5).astype(int) # label is 1 if greater 
                                                     # than 0.5
        c_matrix = confusion_matrix(valid_y, predictions)
        if not os.path.isdir("output/matrices"):
          os.mkdir("output/matrices")
        if context:
            f_name = "output/matrices/"+"c_matrix-"+model+"-"+encoding+"-"+"context.png"
        else:
            f_name = "output/matrices/"+"c_matrix-"+model+"-"+encoding+"-"+".png"
        ax = sns.heatmap(c_matrix, annot=True, fmt=".4g", cmap='Greys')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('True Values ')
        ax.xaxis.set_ticklabels(['False','True'])
        ax.yaxis.set_ticklabels(['False','True'])
        plt.savefig(f_name)
        plt.clf()
        print(classification_report(valid_y, predictions, target_names=labels))
        ml_results[model][encoding]['precision'].append(
            precision_score(valid_y, predictions))
        ml_results[model][encoding]['recall'].append(
            recall_score(valid_y, predictions))
        ml_results[model][encoding]['f1'].append(
            f1_score(valid_y, predictions))
        print(f"\n\nCohen's kappa: {round(100*cohen_kappa_score(valid_y,  predictions),2)}%\n\n")

  return ml_results

def matrix_to_str(matrix):
    matrix_str = ""
    for row in matrix:
        row_str = ""
        for col in row:
            row_str += str(col)+" "
        matrix_str += row_str+"\n"
    return matrix_str
        
def write_latex_table(ml_results, caption, filename):
  latex_table = "\\begin{table}[]\n \\centering\n"+ \
  "\\begin{tabular}{|c|c||c|c|c|c|c|} \\hline\n"+ \
  " model & metric & one-hot & tf-idf & word ngrams & char ngrams & embeddings "+ \
  "\\\\\\hline\n"
  for model in ml_results:
    p, r, f = [], [], []
    for encoding in ml_results[model]:
      if ml_results[model][encoding]['precision'] == []:
        p.append(" & N/A")
        r.append(" & N/A")
        f.append(" & N/A")
      else:
        precision_mean = np.mean(ml_results[model][encoding]['precision'])
        precision_std = np.std(ml_results[model][encoding]['precision'])
        recall_mean = np.mean(ml_results[model][encoding]['recall'])
        recall_std = np.std(ml_results[model][encoding]['recall'])
        f1_mean = np.mean(ml_results[model][encoding]['f1'])
        f1_std = np.std(ml_results[model][encoding]['f1'])
        p.append(" & $"+str(round(precision_mean, 2))+" \\pm "+
                 str(round(precision_std, 2))+" $")
        r.append(" & $"+str(round(recall_mean, 2))+" \\pm "+
                 str(round(recall_std, 2))+" $")
        f.append(" & $"+str(round(f1_mean, 2))+" \\pm "+ \
                 str(round(f1_std, 2))+" $")
    model_name = model.replace('_', ' ')
    latex_table += "\\multirow{3}{*}{"+model_name+"} & precision"+ \
    "".join(p)+"\\\\\n"
    latex_table += " & recall"+"".join(r)+"\\\\\n"
    latex_table += " & f1"+"".join(f)+"\\\\\\hline\n"
  latex_table += "\\end{tabular}\n \\caption{"+caption+ \
  "}\n \\label{tab:my_label}\n \\end{table}\n"
  with open(filename, 'w') as f:
    f.write(latex_table)
  f.close()
  
def plot_results(results, n_iter):
  x = range(1, n_iter+1)
  for model in results:
    for encoding in results[model]:
      if results[model][encoding]['f1'] == []:
        continue
      plt.title(model+" "+encoding)
      plt.plot(x, results[model][encoding]['f1'], label=model+"-"+encoding+".png")
      plt.savefig("output/"+model+"-"+encoding+"-"+datetime.now().strftime('%Y%m%d_%H:%M:%S_')+".png")
      plt.cla()
      plt.clf()

def main():
  train_file = sys.argv[1]
  #val_file = sys.argv[2]
  n_folds = int(sys.argv[2])
  pattern = re.compile(r"(.*/)?(.+)(_.*).csv")
  dataset_name = pattern.search(train_file).group(2)+"-dataset"
  #for b in [False, True]: 
    #if b:
      #title = dataset_name+"-with-context"
    #else:
      #title = dataset_name
    #x_train, y_train, x_val, y_val = extract_datasets(train_file, val_file, b)
  df = get_df(train_file)
  datasets_with_context, datasets_without_context = extract_datasets(train_file) #dataset format: [[xtrain, ytrain, xval, yval]]
  data = [datasets_with_context, datasets_without_context]
  context = True
  for datasets in data:
    if context:
      title = dataset_name+"-with-context"
    else:
      title = dataset_name

    #df = get_df(x_train, y_train, x_val, y_val)        
    #df = get_df(x_train, y_train)
    #datasets = get_splits(n_folds, df)
    #get the encodings
    encoder, count_vect, tfidf_vect, tfidf_vect_ngram, tfidf_vect_ngram_chars, embedded, token = init_encodings(df)
    #get the models
    models = init_models(embedded)
    #train and return the results
    results = train_loop(
    models, datasets, encoder, count_vect, tfidf_vect, tfidf_vect_ngram, tfidf_vect_ngram_chars, embedded, token, context, n_folds)
    plot_results(results, n_folds)
    caption = "Training results for "+title.replace('-', ' ')+"."
    if not os.path.isdir("output"):
      os.mkdir("output")
    filename = "output/"+title+"-"+datetime.now().strftime('%Y%m%d_%H:%M:%S_')+".txt"
    #write results to a text file that can be copied and pasted as a latex table
    write_latex_table(results, caption, filename)
    context = False

if __name__=="__main__":
  main()
  
