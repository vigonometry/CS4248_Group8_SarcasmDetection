import tensorflow as tf
import numpy as np

# Load the saved model
print("Loading the model...")
model = tf.keras.models.load_model('sarcasm_detection_model.keras')

# Function to predict sarcasm for a given text
def predict_sarcasm(text):
    # Make prediction using TensorFlow tensor
    text_input = tf.constant([text])
    prediction = model.predict(text_input)[0][0]
    
    # Convert probability to binary class and get confidence
    is_sarcastic = prediction > 0.5
    confidence = prediction if is_sarcastic else 1 - prediction
    
    return is_sarcastic, confidence

# Test with some sample headlines
sample_headlines = [
    # Known examples from dataset
    "mom starting to fear son's web series closest thing she will have to grandchild",  # Sarcastic
    "former versace store clerk sues over secret black code for minority shoppers",     # Non-sarcastic
    
    # Made-up examples
    "scientists discover new species of animal that communicates entirely through passive-aggressive notes", # Likely sarcastic
    "local temperatures reach record high in summer heat wave",                         # Non-sarcastic news
    "president announces new healthcare policy to take effect next month",              # Non-sarcastic news
    "stock market continues downward trend amid economic concerns",                     # Non-sarcastic news
    "area man only ordered salad to feel better about eating entire cake later",        # Likely sarcastic
    "researchers find drinking coffee daily linked to improved health outcomes",        # Non-sarcastic news
    "new study shows people who believe they are funny are actually insufferable",      # Likely sarcastic
]

# Make predictions
print("\nSarcasm Detection Results:")
print("--------------------------")
for headline in sample_headlines:
    is_sarcastic, confidence = predict_sarcasm(headline)
    
    result = "SARCASTIC" if is_sarcastic else "NOT SARCASTIC"
    print(f"Headline: \"{headline}\"")
    print(f"Prediction: {result} (Confidence: {confidence:.2%})")
    print("--------------------------") 