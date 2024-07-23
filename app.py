from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import MarianMTModel, MarianTokenizer
import torch
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and tokenizer
model_path = 'my_kalenjin_translator'
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

# Function to translate text
def translate_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=128)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Serve the index.html file
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Define the endpoint for translation
@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    text = data['text']
    translated_text = translate_text(text, tokenizer, model)
    return jsonify({'translated_text': translated_text})

# Handle favicon requests
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
