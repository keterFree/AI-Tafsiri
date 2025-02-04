# -*- coding: utf-8 -*-
"""tafsiri2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1c9op6lSAq7L8w4VVMNg7rFZepjsji0_W
"""

# !pip install transformers datasets
# ! pip install -U accelerate
# ! pip install -U transformers

import accelerate
import transformers

import pandas as pd
from google.colab import drive
import ast
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments


transformers.__version__, accelerate.__version__

# prompt: mount drive

from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/upload/engnivkal.txt'

# Read the file contents
with open(file_path, 'r') as file:
  contents = file.read()

print(contents)

# Convert the contents from string to list of lists
data = ast.literal_eval(contents)

# Create a DataFrame
df = pd.DataFrame(data, columns=['English', 'Kalenjin'])

# Remove the '[start]' and '[end]' tags from the Kalenjin column
df['Kalenjin'] = df['Kalenjin'].str.replace('[start]', '').str.replace('[end]', '').str.strip()

df.tail()

df.shape

# from google.colab import files

# df.to_excel('engnivkal.xlsx', index=False)  # Save DataFrame to Excel file

# files.download('engnivkal.xlsx')  # Download the Excel file

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tokenize the data
def preprocess_function(examples):
    inputs = examples['English']
    targets = examples['Kalenjin']
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset

# Split the dataset into train and test sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Set training arguments
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

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

evaluation_results = trainer.evaluate()
print(f"Evaluation results: {evaluation_results}")

# !pip install torch
import torch

test_sentences = [
    "Hello friend.",
    "bear fruit."
]

inputs = tokenizer(test_sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Move inputs to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(inputs["input_ids"], max_length=128)

translations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

for i, translation in enumerate(translations):
    print(f"English: {test_sentences[i]}")
    print(f"Kalenjin: {translation}")

# prompt: save model

trainer.save_model("my_kalenjin_translation_model")
# Save the model and tokenizer
model.save_pretrained("./my_kalenjin_translator")
tokenizer.save_pretrained("./my_kalenjin_translator")

# !pip install shutil
import shutil

# Compress the directory
shutil.make_archive('drive/MyDrive/final/my_kalenjin_translator', 'zip', './my_kalenjin_translator')

from google.colab import files

# Download the file
files.download('/content/my_kalenjin_translation_model')

from google.colab import drive
drive.mount('/content/drive')

import zipfile

# Path to the zip file
zip_path = '/content/drive/MyDrive/final/my_kalenjin_translator.zip'

# Unzip the model
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/drive/MyDrive/final/my_kalenjin_translator')

from transformers import MarianMTModel, MarianTokenizer

# Load the model and tokenizer
model_path = 'my_kalenjin_translator'
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

import torch
# Function to test the model
def translate_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=128)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Example text to translate
example_text = " I have learned to be content whatever the circumstances"

# Get the translation
translated_text = translate_text(example_text, tokenizer, model)
translated_text

