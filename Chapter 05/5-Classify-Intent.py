# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: venv-cookbook
#     language: python
#     name: python3
# ---

# %% 
"""
27. Exploring the CLINC150 dataset
! pip install kagglehub
"""

# %%
# Step 1 - Get the data
import pandas as pd
from kagglehub import (
    dataset_download, 
    load_dataset, 
    KaggleDatasetAdapter,
)

def get_samples(topics=None):

    CLINC150 = "philanoe/clinc150"

    dataset_download(CLINC150)

    # Load a DataFrame with Kaggle data
    ser = load_dataset(
        KaggleDatasetAdapter.PANDAS,
        CLINC150,
        "data/data_full.json",
        pandas_kwargs={
            'typ': 'series', 
            'orient': 'index'
        }
    )

    df = pd.DataFrame(ser.sum()) 
    df.columns = ['text', 'intent']

    # Sample the dataset
    if topics:
        if 'oos' in topics:
            oos = df[df['intent'].isin(['oos'])].iloc[0:150]
            etc = [t for t in topics if t != 'oos']
        else:
            oos = pd.DataFrame()
            etc = topics

        samples = pd.concat([
            df[df['intent'].isin(etc)],
            oos
        ])
    else:
        samples = df

    return samples
    
df = get_samples()
df['words'] = df.text.str.count(' ') + 1
df['length'] = df.text.str.len()

# %%
"""
pip install wordlcloud
pip install matplotlib
"""
from wordcloud import WordCloud
import matplotlib.pyplot as plt

intents = "\n".join(df.intent.unique())

wordcloud = WordCloud(
    width=800, 
    height=400, 
    background_color='white'
).generate(intents)

plt.imshow(
    wordcloud, 
    interpolation='bilinear'
)

plt.axis('off')
plt.show()

# %%
"""
pip install scikit-learn
"""
from sklearn.feature_extraction.text import (
    CountVectorizer
)
from sklearn.metrics.pairwise import (
    cosine_similarity
)

# Vectorize phrases by topic
intents =  df.intent.str.replace("_", " ").unique()
vec = CountVectorizer().fit_transform(intents)

# Compute pairwise cosine similarity
similarity_matrix = cosine_similarity(vec)

plt.imshow(
    similarity_matrix, 
    cmap='gray_r'
)
plt.show()

# %%
# Show topic imbalance
df.groupby('intent').text.count().sort_values()

# %%
# Find duplicate phrases
mask = df.duplicated(subset='text', keep=False)
for t in df[mask].text.unique():
    print(t)

# %%
df.words.plot.hist()

# %%
# Group by topic and calculate average text length. 
avg_len = df.groupby('intent')['words'].mean()
avg_len.sort_values().head(10).plot.barh()

# %% 
"""
28.	Zero and few shot learning 
"""

# %%
# Zero-shot with gpt-4o-mini
from ipython_secrets import get_secret
import openai

openai.api_key = get_secret('OPENAI_API_KEY')
EMBEDDING_MODEL = 'text-embedding-ada-002'
GPT_MODEL = "gpt-4o-mini"
prompt = """
Can you tell me what to do as i am in the airport 
and there is still no sign of my suitcase.
"""
response = openai.chat.completions.create(
  model=GPT_MODEL,
  messages=[{
        "role": "user", 
        "content": prompt
      },
    ]
)
answer = response.choices[0].message.content
print(answer)

# %%
# One-shot with gpt-4o-mini
TRAINING = """
You are a topic classifier. 
Respond with a label showing the intent of the prompt.
"""

PROMPT = """
Can you tell me what to do. 
I'm in the airport & there is no suitcase.
"""
response = openai.chat.completions.create(
  model=GPT_MODEL,
  messages=[
        {"role": "system", "content": TRAINING},
        {"role": "user", "content": PROMPT},
    ]
)
answer = response.choices[0].message.content
print(answer)

# %%
# Few-shot with gpt-4o-mini
TRAINING = """
You are a topic classifier. 
Respond to prompts by labelling the intent based on:

lost_luggage: i lost all my stuff that i had packed
lost_luggage: my luggage seems to have gone missing
travel_alert: is travel to monterrey safe now
travel_alert: safety concerns for malaysia

Classify misses as "oos" (out of scope)".
Respond in YAML format.
"""

PROMPT = [
  "I am in the airport and there is still no sign of my suitcase.",
  "Is it safe to put a battery in my hold luggage?",
  "Can I travel to Scotland?"
]

response = openai.chat.completions.create(
  model=GPT_MODEL,
  messages=[
        {"role": "system", "content": TRAINING},
        {"role": "user", "content": PROMPT[0]},
        {"role": "user", "content": PROMPT[1]},
        {"role": "user", "content": PROMPT[2]},
    ]
)
answer = response.choices[0].message.content

# %%
print(answer)

# %% 
"""
29. Distilbert finely tuned
"""

# %%
# 29.1 Get tokenizer and model

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification as AM4SC
)

REF = "MhF/distilbert-base-uncased-distilled-clinc"
tokenizer = AutoTokenizer.from_pretrained(REF)
model = AM4SC.from_pretrained(REF)

labels = model.config.id2label.values()
print(len(labels), list(labels)[0:3]) 

# %%
# 29.2 Quick look
from torch.nn.functional import softmax

# Input text
text = "I lost my bag at the airport."

# Tokenize input
inputs = tokenizer(
    text, 
    return_tensors="pt", 
    padding=True, 
    truncation=True
)

# Get model outputs
outputs = model(**inputs)
logits = outputs.logits

# Convert logits to probabilities
probs = softmax(logits, dim=1)

# Predicted class
i = probs.argmax(dim=1).item()
pred = model.config.id2label[i]
print(f"Predicted class: {pred}")

# %%
# 29.3 Test a balanced sample
from torch.nn.functional import softmax

topics = ['lost_luggage', 'travel_alert', 'oos']
samples = get_samples(topics)

# Prepare true and predicted labels
y_true = []
y_pred = []

for _, row in samples.iterrows():
    # Tokenize the input text
    inputs = tokenizer(
        row['text'], 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    
    # Get model predictions
    outputs = model(**inputs)
    logits = outputs.logits
    probs = softmax(logits, dim=1)
    
    # Predicted intent
    i = probs.argmax(dim=1).item()
    pred = model.config.id2label[i]
    
    # Append true and predicted labels
    y_true.append(row['intent'])
    y_pred.append(pred)

# %%
# 29.4 Generate and Visualize the Confusion Matrix
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    ConfusionMatrixDisplay
)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=topics)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, 
    display_labels=topics
)
disp.plot(
    cmap="Blues", 
    xticks_rotation="vertical", 
    include_values=True
)

plt.show()

# %%
# 29.5 Generate classification report
from sklearn.metrics import classification_report

report = classification_report(
    y_true, 
    y_pred, 
    target_names=topics, 
    labels=topics
)
print(report)
spam = [y for y in y_pred if y not in topics]
print("Misses", len(spam))

# %%
"""
 29. Stochastic Gradient Descent (SDG)
"""

# %%
# 29.1 Data preparation
from sklearn.model_selection import train_test_split

# Sample the dataset
topics = ['lost_luggage', 'travel_alert', 'oos']
samples = get_samples(topics)

X1, X2, y1, y2 = train_test_split(
    samples['text'], 
    samples['intent'],
    test_size=0.2, 
    random_state=42
)

# %%
# 29.2 Pipeline setup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

sgdc = SGDClassifier(
        max_iter=1000, 
        tol=1e-3
)
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', sgdc)
])

# %%

# 29.3 Train the model on the training set
pipeline.fit(X1, y1)

# %%
# 29.4 Classifying sample inputs

mini_test = [
    "I lost my bag at the airport.",
    "Is it safe to travel to Turkey now?",
    "Can you recommend a good movie?"
]
pred = pipeline.predict(mini_test)
for phrase, topic in zip(mini_test, pred):
    print(f"{topic}: {phrase}")

# %%
# 29.5 Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict the test set labels
pred = pipeline.predict(X2)

# Compute the confusion matrix
cm = confusion_matrix(
    y2, 
    pred, 
    labels=topics
)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, 
    display_labels=topics
)
disp.plot(cmap='Blues')

# Show the plot
import matplotlib.pyplot as plt
plt.show()

# %%
# 29.6 Pickle the pipeline
import joblib

joblib.dump(pipeline, 'models/sdgc_clinc150_pipeline.pkl')
print("Pipeline saved to disk.")

# %%
"""
30. MLView 
"""

# %%
# 30.1 Get samples
from sklearn.model_selection import train_test_split

samples = get_samples()

X1, X2, y1, y2 = train_test_split(
    samples['text'], 
    samples['intent'],
    test_size=0.2, 
    random_state=42
)

# %%
# 30.2 Load the model
import joblib
PKL = "models/sdgc_clinc150_pipeline.pkl"
sgd = joblib.load(PKL)
print("Pipeline loaded from disk.")

# %%
# 30.3 Run an experiment
"""
! pip install mlflow
"""
import time
from mlflow import (
    log_metrics,
    log_input,
    start_run,
    set_experiment,
    get_run
)
from mlflow.data import from_pandas
from mlflow.sklearn import log_model
from mlflow.models.signature import infer_signature

set_experiment("Cookbook")
with start_run(run_name="SGDClassifier") as run:

    log_input(from_pandas(samples))
    log_model(
        sk_model=sgd,
        artifact_path="model",
        signature=infer_signature(samples)
    )

    start_time = time.time()
    pred = sgd.predict(X2)

    log_metrics({
        "time": round(time.time() - start_time,3),
        "accuracy": round(sum(pred == y2)/len(pred),2)
    })

r = get_run(run.info.run_id)
print("run_id", r.info.run_id)
print("metrics", r.data.metrics)



