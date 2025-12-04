"""
Script to predict rating (1-5) from review text using trained RoBERTa model
"""

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

def load_model(model_path='./roberta_rating_model'):
    """
    Load the trained model and tokenizer
    """
    print(f"Loading model from {model_path}...")
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    
    # Set to evaluation mode
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model loaded on {device}")
    return model, tokenizer, device


def predict_rating(text, model, tokenizer, device, max_length=512):
    """
    Predict rating (1-5) for a given review text
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
    
    # Convert class (0-4) to rating (1-5)
    predicted_rating = predicted_class + 1
    
    # Get confidence scores for all ratings
    confidence_scores = {
        rating: prob.item() 
        for rating, prob in enumerate(probabilities[0], start=1)
    }
    
    return predicted_rating, confidence_scores


def main():
    # Load model
    try:
        model, tokenizer, device = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model has been trained first!")
        return
    
    print("\n" + "=" * 60)
    print("Review Rating Predictor")
    print("=" * 60)
    print("\nEnter review text to predict rating (1-5 stars)")
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        # Get input
        review_text = input("Review text: ").strip()
        
        if review_text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not review_text:
            print("Please enter some text.")
            continue
        
        # Predict
        try:
            rating, confidences = predict_rating(review_text, model, tokenizer, device)
            
            print(f"\nPredicted Rating: {rating} stars")
            print("\nConfidence Scores:")
            for r in range(1, 6):
                conf = confidences[r]
                bar = "█" * int(conf * 50)
                print(f"  {r} star{'s' if r > 1 else ''}: {conf*100:5.2f}% {bar}")
            print()
            
        except Exception as e:
            print(f"Error during prediction: {e}\n")


if __name__ == '__main__':
    main()



