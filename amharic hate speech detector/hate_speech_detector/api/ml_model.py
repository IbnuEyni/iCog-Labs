import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Resolve the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "saved_modell")

# Load the model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    raise OSError(f"Error loading the model or tokenizer from {MODEL_DIR}: {e}")

def predict_hate_speech(text):
    """
    Predict if the input text contains hate speech.
    :param text: A string containing the input text.
    :return: A dictionary with the text and the prediction label.
    """
    try:
        # Tokenize input text
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512  # Ensure it doesn't exceed model's max length
        )
        # Get model predictions
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        return {"text": text, "prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

# For testing purposes
if __name__ == "__main__":
    sample_text = "ሰላም እንዴት ንው።"
    result = predict_hate_speech(sample_text)
    print(f"Prediction Result: {result}")
