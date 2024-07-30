
# Kalenjin Translation Model Mini-Project
## Group Members
- Titus Keter - CS/MG/2163/09/21
- Paul Ruoya - CS/MG/2595/09/21
- Ezekiel Kiprotich - CS/MG/1930/09/21

## Documentation Summary
### Objective
The project aims to create and deploy a machine learning model that translates English text to Kalenjin using the `transformers` library, particularly the Helsinki-NLP models.

### Data Acquisition
The dataset is mainly made up of English King James Version Bible and a Kalenjin bible stored in `engnivkal.txt` file.
i.e.
structure: An Englis verse is followed by te Kalenjin version enclosed by `[start]` and `[end]`.

### Data Cleaning
Tags `[start]` and `[end]` were removed from the Kalenjin translations, and the cleaned data was converted into a pandas DataFrame.

### Exploratory Data Analysis (EDA)
Basic EDA was conducted to understand the data structure and distribution and check for inconsistencies.

### Feature Engineering
The DataFrame was converted to a Hugging Face `Dataset` object for efficient tokenization and preprocessing. English and Kalenjin texts were tokenized.

### Model Building
The `Helsinki-NLP/opus-mt-en-mul` model was used. The data was tokenized and split into training and testing sets. Training involved five epochs with evaluation at each epoch.

### Model Evaluation
The model's performance was evaluated using standard sequence-to-sequence metrics, and the results were assessed for accuracy and fluency.
```
Evaluation results: {'eval_loss': 0.5533947944641113, 'eval_runtime': 10.0125, 'eval_samples_per_second': 303.621, 'eval_steps_per_second': 18.976, 'epoch': 5.0}
```
### Deployment
The trained model and tokenizer were saved and deployed as a Flask web application, allowing users to input English text and receive Kalenjin translations.

### Key Steps and Code
1. **Data Loading and Cleaning**: Loaded and cleaned data from Google Drive, converted it into a DataFrame.
2. **Data Preparation**: Converted DataFrame to Hugging Face Dataset, loaded tokenizer and model, tokenized the data.
3. **Model Training**: Split the dataset, defined training arguments, initialized and trained the model.
4. **Model Evaluation**: Evaluated the model and printed results.
5. **Save and Deploy**: Saved the model and tokenizer, deployed as a Flask app.

### Conclusion
This project demonstrates the entire process of building and deploying a language translation model, serving as a guide for similar projects.

