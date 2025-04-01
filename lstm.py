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

# Define the unidirectional LSTM model
class UniRNN(nn.Module):
    def __init__(self, input_size=100, hidden_size=128, num_layers=2, dropout=0.2):
        super(UniRNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, (hidden, _) = self.lstm(x)
        # Take the output from the last time step
        out = hidden[-1, :, :]  # Shape: (batch_size, hidden_size)
        out = self.dropout(out)
        out = self.fc(out)
        return torch.sigmoid(out)

# Define the bidirectional LSTM model
class BiRNN(nn.Module):
    def __init__(self, input_size=100, hidden_size=128, num_layers=2, dropout=0.2):
        super(BiRNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        # Note: hidden_size * 2 because of bidirectional
        self.fc = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, (hidden, _) = self.lstm(x)
        
        # In bidirectional LSTM, hidden contains both forward and backward final states
        # Concatenate the final states from both directions
        # hidden shape: (num_layers * num_directions, batch_size, hidden_size)
        # We want the last layer's hidden state for both directions
        forward = hidden[-2, :, :]  # Shape: (batch_size, hidden_size)
        backward = hidden[-1, :, :]  # Shape: (batch_size, hidden_size)
        out = torch.cat((forward, backward), dim=1)  # Shape: (batch_size, hidden_size*2)
        
        out = self.dropout(out)
        out = self.fc(out)
        return torch.sigmoid(out)

# Training function
def train_model(model, X_train, y_train, X_val, y_val, criterion, optimizer, num_epochs=10, batch_size=64):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    n_samples = X_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        # Process in batches
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * (end_idx - start_idx)
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == y_batch).sum().item()
        
        # Calculate epoch statistics
        train_loss /= n_samples
        train_acc = train_correct / n_samples
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            outputs = model(X_val)
            loss = criterion(outputs, y_val)
            val_loss = loss.item()
            predicted = (outputs > 0.5).float()
            val_correct = (predicted == y_val).sum().item()
        
        val_acc = val_correct / X_val.shape[0]
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        print('-' * 50)
    
    return model, train_losses, val_losses, train_accs, val_accs

# Evaluate model
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = (outputs > 0.5).float()
        
        # Convert tensors to numpy for sklearn metrics
        y_true = y_test.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        
        # Print classification report
        print(classification_report(y_true, y_pred))
        
        # Calculate accuracy
        accuracy = (predicted == y_test).sum().item() / y_test.shape[0]
        print(f'Test Accuracy: {accuracy:.4f}')
    
    return accuracy, y_pred

# Initialize models
print("Initializing models...")
input_size = 100  # Word2Vec embedding dimension
hidden_size = 128

# Unidirectional LSTM
uni_rnn = UniRNN(input_size, hidden_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(uni_rnn.parameters(), lr=0.001)

# Train unidirectional 
print("\nTraining Unidirectional LSTM...")
uni_rnn, uni_train_losses, uni_val_losses, uni_train_accs, uni_val_accs = train_model(
    uni_rnn, X_train, y_train, X_test, y_test, criterion, optimizer, num_epochs=10, batch_size=64
)

# Evaluate unidirectional LSTM
print("\nEvaluating Unidirectional LSTM...")
uni_accuracy, uni_predictions = evaluate_model(uni_rnn, X_test, y_test)

# Bidirectional LSTM
bi_rnn = BiRNN(input_size, hidden_size).to(device)
optimizer = optim.Adam(bi_rnn.parameters(), lr=0.001)

# Train bidirectional LSTM
print("\nTraining Bidirectional LSTM...")
bi_rnn, bi_train_losses, bi_val_losses, bi_train_accs, bi_val_accs = train_model(
    bi_rnn, X_train, y_train, X_test, y_test, criterion, optimizer, num_epochs=10, batch_size=64
)

# Evaluate bidirectional LSTM
print("\nEvaluating Bidirectional LSTM...")
bi_accuracy, bi_predictions = evaluate_model(bi_rnn, X_test, y_test)

# Compare models
print("\nModel Comparison:")
print(f"Unidirectional LSTM Accuracy: {uni_accuracy:.4f}")
print(f"Bidirectional LSTM Accuracy: {bi_accuracy:.4f}")