from transformers import BertTokenizer, BertModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

# Example text
text = "Here is some text to encode"

# Encode text
encoded_input = tokenizer(text, return_tensors='pt')

# Process with the model
output = model(**encoded_input)

# Output the result
print(output)
