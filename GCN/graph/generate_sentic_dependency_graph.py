import numpy as np
import spacy
import pickle
from tqdm import tqdm
import pandas as pd
import ast

nlp = spacy.load('en_core_web_sm')

def load_sentic_word():
    path = './senticNet/senticnet_word.txt'
    senticNet = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = sentic
    fp.close()
    return senticNet

def sentic_dependency_adj_matrix(text, senticNet):
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    
    for token in document:
        if str(token) in senticNet:
            sentic = float(senticNet[str(token)]) + 1
        else:
            sentic = 0

        if token.i < seq_len:
            matrix[token.i][token.i] = 1 * sentic
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1 * sentic
                    matrix[child.i][token.i] = 1 * sentic
    return matrix


def safe_literal_eval(val):
    if isinstance(val, str) and val != 'nan':
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []  # Return an empty list in case of an invalid string
    return []  # Return an empty list for NaN or non-string values

def process_sdat(filename, col, export_file_name):
    senticNet = load_sentic_word()
    fin = pd.read_csv(filename)
    fin[col] = fin[col].apply(safe_literal_eval)  # Apply the safe literal eval function
    fin['text'] = fin[col].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')  # Handle case where x might not be a list
    lines = fin['text'].tolist()
    #fin.close()
    idx2graph = {}
    fout = open(export_file_name+'.graph_sdat', 'wb')
    print('Generating sentic dependancy graph...')
    for i in tqdm((range(0, len(lines)))):
        text = lines[i].lower().strip()
        adj_matrix = sentic_dependency_adj_matrix(text, senticNet)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    print('Done!')
    fout.close()
