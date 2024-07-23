## Documentation for Kalenjin Translation Model Mini-Project

### Objective
The goal of this project is to build and deploy a machine learning model that translates English text to Kalenjin. This involves training a sequence-to-sequence model using the `transformers` library, specifically leveraging the Helsinki-NLP models for multi-lingual translation, and deploying the trained model as a web application.

### Data Acquisition
The dataset used for this project is a collection of English sentences paired with their Kalenjin translations. This dataset is stored in a file named `engnivkal.txt` located in a Google Drive folder. The file contents were read and processed into a list of lists format.

### Data Cleaning
The Kalenjin translations in the dataset contained tags `[start]` and `[end]`, which were removed to clean the data. The cleaned dataset was then converted into a pandas DataFrame for further processing.

### Exploratory Data Analysis (EDA)
Basic EDA was performed to understand the structure and distribution of the data. The dataset was inspected for any inconsistencies or irregularities. The shape of the DataFrame was checked to ensure it contained the expected number of rows and columns.

### Feature Engineering
The cleaned DataFrame was converted into a Hugging Face `Dataset` object. This allowed for efficient tokenization and preprocessing using the `transformers` library. The English and Kalenjin text data were tokenized to prepare them for model training.

### Model Building
The model selected for this project is the `Helsinki-NLP/opus-mt-en-mul` model, a multi-lingual sequence-to-sequence model available from the Hugging Face model hub. The model and tokenizer were loaded, and the data was tokenized using the model's tokenizer. The tokenized dataset was then split into training and testing sets.

Training arguments were defined using `Seq2SeqTrainingArguments`, and a `Seq2SeqTrainer` was initialized with the model, training arguments, and datasets. The model was trained for five epochs, with evaluation performed at each epoch to monitor performance.

### Model Evaluation
The trained model was evaluated using standard sequence-to-sequence evaluation metrics. The evaluation results were printed to assess the model's performance on the test set.

### Results Interpretation
The model was tested with a few English sentences to translate them into Kalenjin. The results showed the model's capability to produce meaningful translations. These translations were inspected for accuracy and fluency.

### Documentation and Presentation
The entire process, from data preprocessing to model training and evaluation, was documented. The following sections provide a detailed walkthrough of the steps taken and the code used.

### Deployment
The trained model and tokenizer were saved and deployed as a Flask web application. The Flask app serves an HTML front-end that allows users to input English text and receive Kalenjin translations. The app handles CORS to ensure cross-origin requests are permitted.

### Code and Steps
Below are the detailed steps and code snippets used throughout the project:

#### 1. Data Loading and Cleaning
```python
from google.colab import drive
import ast
import pandas as pd

# Mount Google Drive
drive.mount('/content/drive')

# Load data from Google Drive
file_path = '/content/drive/MyDrive/upload/engnivkal.txt'
with open(file_path, 'r') as file:
    contents = file.read()

# Convert contents to list of lists
data = ast.literal_eval(contents)

# Create DataFrame
df = pd.DataFrame(data, columns=['English', 'Kalenjin'])

# Clean Kalenjin column
df['Kalenjin'] = df['Kalenjin'].str.replace('[start]', '').str.replace('[end]', '').str.strip()

# Display cleaned DataFrame
df.tail()
```

#### 2. Data Preparation
```python
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-mul")

# Tokenization function
def preprocess_function(examples):
    inputs = examples['English']
    targets = examples['Kalenjin']
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)
```

#### 3. Model Training
```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Split dataset into training and testing sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=5,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=True,
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()
```

#### 4. Model Evaluation
```python
# Evaluate the model
evaluation_results = trainer.evaluate()
print(f"Evaluation results: {evaluation_results}")
```

#### 5. Save and Deploy the Model
```python
# Save the model and tokenizer
model.save_pretrained("./my_kalenjin_translator")
tokenizer.save_pretrained("./my_kalenjin_translator")

# Compress the directory
import shutil
shutil.make_archive('drive/MyDrive/final/my_kalenjin_translator', 'zip', './my_kalenjin_translator')

# Download the file (optional)
from google.colab import files
files.download('/content/my_kalenjin_translation_model')
```

### Flask App for Deployment
```python
# app.py
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
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Conclusion
This project demonstrates the end-to-end process of building and deploying a machine learning model for language translation. By following the steps outlined in this documentation, you can replicate the process to train and deploy your own models for various applications.