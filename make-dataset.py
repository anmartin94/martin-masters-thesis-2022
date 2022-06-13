"""
Author: Anna Martin
Description: 
This program extracts 4 .csv files from the annotated NLPSharedTasks corpus:
1) A training set containing full paper texts
2) A test set containing full paper texts
3) A reduced training set containing only sections that have a positive sample
4) A reduced test set containing only sections that have a positive sample
Usage: 
The name of the folder containing the annotated .txt corpus files must be passed
as a command line argument, as well as the name of the folder the output csv
files should be stored in.
EXAMPLE: python3 make-dataset.py shared-task-corpus/annotated-txt-files/ shared-task-classification-project/datasets/
"""

import numpy as np
import os
import pandas as pd
import re
import stanza
import sys

def extract_data_into_dict(annotated_data_path):
    stanza.download('en', processors='tokenize')
    nlp = stanza.Pipeline(lang='en', processors='tokenize')

    ann_files = os.listdir(annotated_data_path)

    processed_data = {}
    
    paper_ids = []

    sentence_id = 0
    for ann_file in ann_files:
        print("Progress:", ann_files.index(ann_file), "/", len(ann_files)) #script might take between 10 and 12 minutes to run
        sentences = []
        paper_id = ann_file[10:-4]
        paper_ids.append(paper_id)
        lines = open(annotated_data_path+ann_file)
        next_header = True
        new_header = True
        header = "title"
        n_sentences_in_paper = 0
        for line in lines: #each line contains a single paragraph
            if next_header == True:
                header = line[:-1]
                next_header = False
                continue
            if line == "abstract\n":
                header = "abstract"
                continue
            if line != '\n':
                doc = nlp(line)
                n_sentences_in_paper += len(doc.sentences)
                for sentence in enumerate(doc.sentences):
                    if new_header:
                        sentences.append([])
                        new_header = False
                    sentences[len(sentences)-1].append(sentence[1].text)
                    processed_data[len(processed_data)] = {'paper_id': paper_id, 'header': header}
            else: #an empty line indicates a section break; the next line will be the section header
                next_header = True
                new_header = True
                continue 
        global_pos = 0
        for section in sentences:
            found_task = False
            for local_pos in range(len(section)):
            #calculate the positional information for each sample
                local_pct, global_pct = 0, 0
                local_ratio = local_pos/len(section)
                global_ratio = global_pos/n_sentences_in_paper
                if local_ratio < 0.25:
                    local_pct = 1
                elif local_ratio < 0.5:
                    local_pct = 2
                elif local_ratio < 0.75:
                    local_pct = 3
                else:
                    local_pct = 4
                if global_ratio < 0.25:
                    global_pct = 1
                elif global_ratio < 0.5:
                    global_pct = 2
                elif global_ratio < 0.75:
                    global_pct = 3
                else:
                    global_pct = 4
                    
                processed_data[sentence_id]['local_pos'] = local_pos
                processed_data[sentence_id]['global_pos'] = global_pos
                processed_data[sentence_id]['local_pct'] = local_pct
                processed_data[sentence_id]['global_pct'] = global_pct
                found_end = False
                if "<TASK>" in section[local_pos]:
                    found_task = True
                if "</TASK>" in section[local_pos]:
                    found_end = True
                section[local_pos] = section[local_pos].replace("<TASK>", "")
                section[local_pos] = section[local_pos].replace("</TASK>", "")
                processed_data[sentence_id]['sentence'] = section[local_pos]
                if found_task:
                    processed_data[sentence_id]['label'] = 1
                else:
                    processed_data[sentence_id]['label'] = 0
                if found_end:
                    found_task = False
                sentence_id += 1
                global_pos += 1       
    return processed_data, list(set(paper_ids))

def flatten_dict(processed_data):
    flattened_data = dict.fromkeys(['sentence_id', 'paper_id', 'header', 'local_pos', 'global_pos', 'local_pct', 'global_pct', 'sentence', 'label'])
    for record in flattened_data:
        flattened_data[record] = []
    for record in processed_data:
        flattened_data['sentence_id'].append(record)
        flattened_data['paper_id'].append(processed_data[record]['paper_id'])
        flattened_data['header'].append(processed_data[record]['header'])
        flattened_data['local_pos'].append(processed_data[record]['local_pos'])
        flattened_data['global_pos'].append(processed_data[record]['global_pos'])
        flattened_data['local_pct'].append(processed_data[record]['local_pct'])
        flattened_data['global_pct'].append(processed_data[record]['global_pct'])
        flattened_data['sentence'].append(processed_data[record]['sentence'])
        flattened_data['label'].append(processed_data[record]['label'])
        
    return flattened_data
    
def reduce_data(processed_data, df, pids):
    for paper_id in pids:
        sections_to_keep = []
        sections_to_drop = []
        for i in range(len(processed_data['paper_id'])):
            if processed_data['paper_id'][i] != paper_id:
                continue
            if processed_data['label'][i] == 1:
                sections_to_keep.append(processed_data['header'][i])
        for header in processed_data['header']:
            if header not in sections_to_keep:
                sections_to_drop.append(header)
        sections_to_keep = list(set(sections_to_keep))
        sections_to_drop = list(set(sections_to_drop))
        if len(sections_to_keep) == 1 and sections_to_keep[0] == "title":
            sections_to_keep.append("abstract")
            sections_to_drop.remove("abstract")
        for section in sections_to_drop:
            paper_idx_to_drop = df[df['paper_id'] == paper_id].index
            header_idx_to_drop = df[df['header'] == section].index
            idx_to_drop = set.intersection(set(paper_idx_to_drop), set(header_idx_to_drop))
            df.drop(idx_to_drop, inplace=True)     
    return df
    
    
                  
def organize_data_into_csv(processed_data, all_paper_ids, output_path):
    test_paper_ids = np.random.choice(all_paper_ids, 26, replace=False)
    test_data = {}
    sentence_idx_to_pop = []
    for sentence in processed_data:
        if processed_data[sentence]['paper_id'] in test_paper_ids:
            sentence_idx_to_pop.append(sentence)
    for idx in sentence_idx_to_pop:
        test_data[idx] = processed_data.pop(idx)
    processed_data = flatten_dict(processed_data)
    test_data = flatten_dict(test_data)
    full_train_df = pd.DataFrame.from_dict(processed_data)
    full_test_df = pd.DataFrame.from_dict(test_data)
    reduced_train_df = full_train_df.copy()
    reduced_test_df = full_test_df.copy()
    reduced_train_df = reduce_data(processed_data, reduced_train_df, list(set(full_train_df['paper_id'].tolist())))
    reduced_test_df = reduce_data(test_data, reduced_test_df, list(set(full_test_df['paper_id'].tolist())))
    full_train_df.to_csv(output_path+"full_train.csv", sep='\t')
    full_test_df.to_csv(output_path+"full_test.csv", sep='\t')
    reduced_train_df.to_csv(output_path+"reduced_train.csv", sep='\t')
    reduced_test_df.to_csv(output_path+"reduced_test.csv", sep='\t')
    
    
def main():
    annotated_data_path = sys.argv[1]
    output_path = sys.argv[2]
    processed_data, all_paper_ids = extract_data_into_dict(annotated_data_path) 
    organize_data_into_csv(processed_data, all_paper_ids, output_path)

if __name__=="__main__":
  main()
            
            
            
        
        
            
        
        

