import pandas as pd
from sklearn.model_selection import train_test_split as sklearn_train_test_split


def train_test_split(data: pd.DataFrame, test_size):
    # Split the data into training and test sets and save them as csv files
    train, test = sklearn_train_test_split(data, test_size=test_size, random_state=42)
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)
    
    # save to train.jsonl and test.jsonl
    train.to_json('train.jsonl', orient='records', lines=True)
    test.to_json('test.jsonl', orient='records', lines=True)
    
df = pd.read_csv('data/raw.csv')
train_test_split(df, test_size=0.2)