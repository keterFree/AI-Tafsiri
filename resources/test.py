from transformers import MarianMTModel, MarianTokenizer
import torch

# Load the model and tokenizer
model_path = 'my_kalenjin_translator'
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

# Function to test the model
def translate_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=128)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Example text to translate
example_text = "I have learned to be content whatever the circumstances"

# Get the translation
translated_text = translate_text(example_text, tokenizer, model)
print(f'translated_text: {translated_text}')
