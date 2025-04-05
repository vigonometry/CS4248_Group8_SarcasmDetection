import numpy as np
import spacy
import pickle
from tqdm import tqdm
import pandas as pd
import ast

nlp = spacy.load('en_core_web_sm')

def gen_adj_matrix(text):
    document = nlp(text)
    seq_len = len(text.split())
    
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1

    return matrix

def safe_literal_eval(val):
    """Safely convert string representations of lists into actual lists."""
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else val
    except (ValueError, SyntaxError):
        return val  # If conversion fails, return the original value

def process_adj_matrix(filename, col, export_file_name):
    fin = pd.read_csv(filename)
    fin[col] = fin[col].apply(safe_literal_eval)  # Apply the safe literal eval function
    if fin[col].apply(lambda x: isinstance(x, list)).all():
        fin["text"] = fin[col].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    else:
        fin['text'] = fin[col]
    print(fin['text'][:2])
    lines = fin['text'].tolist()
    #lines = fin.readlines()
    #fin.close()
    idx2graph = {}
    fout = open(export_file_name+'.graph', 'wb')
    print('Generating adjacency matrix...')
    for i in tqdm(range(0, len(lines))):
        text = lines[i].lower().strip()
        adj_matrix = gen_adj_matrix(text)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    print('Done!')        
    fout.close()