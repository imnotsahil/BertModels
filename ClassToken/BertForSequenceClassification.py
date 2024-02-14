#from transformers import BertTokenizer, BertForSequenceClassification
#import torch
#
## Function to classify text
#def classify_text(text, model, tokenizer):
#    # Encode the text
#    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#    
#    # Predict
#    with torch.no_grad():
#        logits = model(**inputs).logits
#
#    # Interpret the result
#    predictions = torch.nn.functional.softmax(logits, dim=-1)
#    labels = ['Negative', 'Positive']
#    label = labels[predictions.argmax()]
#    
#    return label
#
## Load pre-trained model and tokenizer
#model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
## Example texts
#texts = ["I love this product!", "I hate this service!"]
#
## Classify each text
#for text in texts:
#    result = classify_text(text, model, tokenizer)
#    print(f"Text: '{text}' is classified as {result}")


from transformers import BertTokenizer, BertForSequenceClassification
import torch

def classify_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.nn.functional.softmax(logits, dim=-1)
    # The model uses 5 classes for sentiment
    labels = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
    label = labels[predictions.argmax()]
    return label

# Load a pre-trained and fine-tuned model and tokenizer
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Example texts
texts = ["I love this product!", "I hate this service!"]

# Classify each text
for text in texts:
    result = classify_text(text, model, tokenizer)
    print(f"Text: '{text}' is classified as {result}")

