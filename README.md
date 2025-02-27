# Disaster Tweet Classification Using DistilBERT

## Overview
The goal of this competition is to build a machine learning model that predicts which tweets are about real disasters and which ones aren't. The dataset consists of 10,000 hand-classified tweets.

This notebook utilizes the **DistilBERT** pretrained model from **KerasNLP** for classification.

## About BERT & DistilBERT
**BERT (Bidirectional Encoder Representations from Transformers)** is a Transformer-based model that processes text bidirectionally, considering both left and right context. Pretrained on a large corpus, BERT can be fine-tuned for specific NLP tasks.

**DistilBERT** is a smaller, faster, and lighter version of BERT. It retains 97% of BERT’s language understanding capabilities while being 60% faster and 40% smaller.

## Steps in This Notebook
- Load the dataset
- Explore the dataset
- Preprocess the data
- Load and fine-tune DistilBERT from KerasNLP
- Train and evaluate the model
- Generate predictions

## Dataset
The dataset consists of:
- **id**: Unique identifier for the tweet
- **keyword**: A keyword from the tweet (may be blank)
- **location**: The location of the tweet (may be blank)
- **text**: The actual tweet
- **target**: 1 if the tweet is a real disaster, 0 otherwise

### Load the Data
```python
import pandas as pd

# Load the datasets
df_train = pd.read_csv("/content/drive/MyDrive/nlp-getting-started/train.csv")
df_test = pd.read_csv("/content/drive/MyDrive/nlp-getting-started/test.csv")

# Print dataset information
print(f'Training Set Shape: {df_train.shape}')
print(f'Training Set Memory Usage: {df_train.memory_usage().sum() / 1024**2:.2f} MB')
print(f'Test Set Shape: {df_test.shape}')
print(f'Test Set Memory Usage: {df_test.memory_usage().sum() / 1024**2:.2f} MB')
```

## Data Cleaning & Exploration

### Handling Missing Values and Duplicates
```python
# Fill missing values
df_train['keyword'].fillna('Unknown', inplace=True)
df_train['location'].fillna('Unknown', inplace=True)
df_test['keyword'].fillna('Unknown', inplace=True)
df_test['location'].fillna('Unknown', inplace=True)

# Remove duplicates
df_train.drop_duplicates(subset='text', inplace=True)
df_test.drop_duplicates(subset='text', inplace=True)
```

### Text Cleaning
```python
import re

def clean_text(text):
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove links
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    return text

# Apply text cleaning
df_train['text'] = df_train['text'].apply(clean_text)
df_test['text'] = df_test['text'].apply(clean_text)
```

### Exploratory Data Analysis (EDA)
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Target variable distribution
sns.countplot(x='target', data=df_train)
plt.title('Distribution of Target Variable')
plt.show()

# Text length analysis
df_train['text_length'] = df_train['text'].apply(len)
sns.histplot(data=df_train, x='text_length', hue='target', bins=30, kde=True)
plt.title('Text Length Distribution by Target')
plt.show()
```

## Data Preprocessing
```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

# Extract features and target
X = df_train['text']
y = df_train['target']

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Apply SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_vectorized, y)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
```

## Load DistilBERT Model
```python
import keras_nlp

# Load DistilBERT model
preset = "distil_bert_base_en_uncased"
preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(preset, sequence_length=160)
classifier = keras_nlp.models.DistilBertClassifier.from_preset(preset, preprocessor=preprocessor, num_classes=2)

classifier.summary()
```

## Model Training
```python
import tensorflow as tf

# Convert sparse matrix to dense format
X_train_dense = X_train.toarray()
X_val_dense = X_val.toarray()

# Compile and train the model
classifier.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.fit(X_train_dense, y_train, validation_data=(X_val_dense, y_val), epochs=2, batch_size=32)
```

## Generate Predictions
```python
X_test_vectorized = vectorizer.transform(df_test['text'])
X_test_dense = X_test_vectorized.toarray()

y_pred = classifier.predict(X_test_dense)
y_pred_classes = tf.argmax(y_pred, axis=1)

# Create submission file
submission = pd.DataFrame({"id": df_test["id"], "target": y_pred_classes.numpy()})
submission.to_csv("submission.csv", index=False)
```

## Summary
- Loaded and cleaned the disaster tweet dataset
- Explored data and handled missing values & duplicates
- Preprocessed text using TF-IDF and handled class imbalance with SMOTE
- Loaded a pretrained DistilBERT model and fine-tuned it
- Trained the model and generated predictions for submission
