import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the preprocessed dataset
print("Loading preprocessed dataset...")
df = pd.read_csv('data/preprocessed.csv')
print(f"Dataset shape: {df.shape}")

# Basic dataset analysis
print(f"Label distribution: \n{df['is_sarcastic'].value_counts()}")

# Use headline_cleaned for text input
sentences = df['headline_cleaned'].values
labels = df['is_sarcastic'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    sentences, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Create a simple vectorizer for text
print("Creating text vectorization layer...")
MAX_FEATURES = 10000
SEQUENCE_LENGTH = 50

# Text vectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_FEATURES,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)

# Adapt the layer to the training data
print("Adapting vectorizer to training data...")
vectorize_layer.adapt(X_train)

# Create the model
def create_model():
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
    x = vectorize_layer(inputs)
    
    # Embedding layer
    x = tf.keras.layers.Embedding(MAX_FEATURES + 1, 128)(x)
    
    # Bidirectional LSTM for sequence processing (similar to BERT's bidirectionality)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
    
    # Dense layers for classification
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and train the model
print("Creating model...")
model = create_model()
print(model.summary())

# Use smaller batch size and fewer epochs for faster training
batch_size = 64
epochs = 5

print("Training model...")
history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1
)

# Evaluate the model
print("Evaluating model...")
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Classification Report:\n{report}")

# Save the model
model.save("sarcasm_detection_model.keras")
print("Model saved to 'sarcasm_detection_model.keras'.") 