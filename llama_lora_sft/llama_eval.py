import json
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = 'mistral7B'

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    df.to_csv(f'llama_lora_sft\predictions_{model}.csv', index=False)
    return pd.DataFrame(data)

def calculate_metrics(df):
    y_true = df['is_sarcastic'].apply(lambda x: str(x).lower()[0] == 's').astype(int)
    y_pred = df['prediction'].apply(lambda x: str(x).lower()[0] == 's').astype(int)
    
    # print(y_true)
    
    zero_df = df[df['is_sarcastic'] == 0]
    one_df = df[df['is_sarcastic'] == 1]

    
    print("Number of 0s in y_true:", len(y_true[y_true == 0]))
    print("Number of 1s in y_true:", len(y_true[y_true == 1]))
    print("Number of 0s in y_pred:", len(y_pred[y_pred == 0]))
    print("Number of 1s in y_pred:", len(y_pred[y_pred == 1]))
    
    # Confusion matrix
    confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print("Confusion Matrix:")
    print(confusion_matrix)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

def main():
    file_path = f'llama_lora_sft/predictions_{model}.jsonl'
    
    df = load_jsonl(file_path)
    metrics = calculate_metrics(df)
    
    for metric, value in metrics.items():
        print(f'{metric}: {value:.4f}')

if __name__ == '__main__':
    main()
