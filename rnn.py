import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils

# Load data
spacy_df = pd.read_csv('data/features_nltk.csv')
spacy_df = spacy_df[['headline', 'headline_cleaned', 'tokenized_text_nltk', 'lemmatized_text_nltk', 'is_sarcastic']]

# Process text columns
spacy_df['tokenized_text_nltk'] = spacy_df['tokenized_text_nltk'].apply(lambda x: x.replace('[','').replace(']','').replace("'",'').replace(",",'').split())
spacy_df['lemmatized_text_nltk'] = spacy_df['lemmatized_text_nltk'].apply(lambda x: x.replace('[','').replace(']','').replace("'",'').replace(",",'').split())

tokenized_sentences = spacy_df['tokenized_text_nltk'].tolist()
lemmatized_sentences = spacy_df['lemmatized_text_nltk'].tolist()

# Train Word2Vec models
print("Training Word2Vec models...")
w2v_tokenized = Word2Vec(tokenized_sentences, vector_size=100, min_count=1, workers=4)
w2v_lemmatized = Word2Vec(lemmatized_sentences, vector_size=100, min_count=1, workers=4)

# Function to get Word2Vec embeddings for a sentence
def get_word2vec_embeddings(model, sentence, embed_dim=100):
    embeddings = []
    for word in sentence:
        if word in model.wv:
            # Get and normalize the vector
            vec = model.wv[word]
            # Optional: normalize for better training stability
            vec = vec / (np.linalg.norm(vec) + 1e-6)
            embeddings.append(vec)
        else:
            embeddings.append(np.zeros(embed_dim))
    return torch.tensor(embeddings, dtype=torch.float32)

# Get word embeddings and pad sequences
print("Getting and padding word embeddings...")
sentence_embeddings = [get_word2vec_embeddings(w2v_tokenized, sent) for sent in tokenized_sentences]

# Pad sequences to max length
max_len = max(len(seq) for seq in sentence_embeddings)
padded_embeddings = [torch.cat((seq, torch.zeros(max_len - len(seq), 100))) if len(seq) < max_len else seq for seq in sentence_embeddings]

print(padded_embeddings[0].shape)  # Shape: (max_len, embedding_dim)
print(padded_embeddings[1])
# Convert to tensor
X = torch.stack(padded_embeddings)  # Shape: (num_samples, max_len, embedding_dim)
y = torch.tensor(spacy_df['is_sarcastic'].values, dtype=torch.float32).unsqueeze(1)  # Shape: (num_samples, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

# Modified RNN to handle packed sequences
class RNNClassifier(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=100, num_layers=1):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x is a PackedSequence
        output, (h_n, c_n) = self.rnn(x)
        # Use the last hidden state
        return self.fc(h_n[-1])

# Initialize model, loss function, optimizer
model = RNNClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
print("Training Unidirectional RNN...")
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    
    outputs = model(X_train).squeeze(1)
    loss = criterion(outputs, y_train.squeeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Evaluation
print("Evaluating RNN...")
model.eval()
with torch.no_grad():
    outputs = model(X_test).squeeze(1)
    preds = torch.round(torch.sigmoid(outputs))
    print(classification_report(y_test.cpu().numpy(), preds.cpu().numpy()))
