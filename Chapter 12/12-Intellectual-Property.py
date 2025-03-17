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

# %% [markdown]
# # 63. Patent Summarisation - HTML BS4

# %%
# 63.1 PATENT RETRIEVAL

import requests

# Fetch the patent content from Google Patents
url = "https://patents.google.com/patent/US4959281A/en"
raw = requests.get(url)

raw.status_code, raw.text[0:40], len(raw.text)

# %%
# 63.2 TEXT EXTRACTION

from bs4 import BeautifulSoup

soup = BeautifulSoup(raw.content, "html.parser")

# Extract the main text of the patent
patent_text = soup.find("section", itemprop="description").get_text(
    separator=" ", strip=True
)

patent_text[0:100]

# %%
# 63.3 SUMMARISATION

from ipython_secrets import get_secret
from openai import OpenAI
from IPython.display import display, Markdown

KEY = get_secret("OPENAI_API_KEY")
client = OpenAI(api_key=KEY)

INSTRUCTION = """
You are a highly skilled AI trained in patent comprehension and summarization. 
I would like you to read the given patent description and summarize it into a concise abstract paragraph. 
Aim to retain the most important points, providing a coherent and readable summary 
that could help a person understand the main points without needing to read the entire text. 
Avoid unnecessary details or tangential points."
"""

response = client.chat.completions.create(
    model="gpt-4",
    temperature=0,
    messages=[
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": patent_text},
    ],
)

summary = response.choices[0].message.content
print("\n\n=== SUMMARY ===\n")
display(Markdown(summary))

# %% [markdown]
# # 64. Pythonic patent summarisation
#

# %%
"""
! pip install patent-client
"""
from patent_client import Patent
from IPython.display import display, Markdown

patent = Patent.objects.get("4959281")
display(Markdown(patent.description))

# %% [markdown]
# # 65. Analysing USPTO trademark search results
#

# %%
# 65.1 GET USPTO DATA
import requests
from bs4 import BeautifulSoup

URL = "https://assignment-api.uspto.gov/trademark/advancedSearch"
query = {"fields": "main", "ownerName": "Tesla"}

response = requests.get(URL, params=query, verify=False)
soup = BeautifulSoup(response.text, "xml")

# %%
# 65.2 TRANSFORM THE XML RECORDS
trademark_assignments = []
for doc in soup.find_all("doc"):
    
    # Find the markDesc <arr> tag
    arr_tag = doc.find("arr", {"name": "markDesc"})

    # Extract text from all <str> tags within the <arr>
    if arr_tag:
        descriptions = [str_tag.text for str_tag in arr_tag.find_all("str")]
    else:
         descriptions = []

    record = {
        "id": doc.find("str", {"name": "id"}).text,
        "recordedDate": doc.find("date", {"name": "recordedDate"}).text,
        "conveyanceText": doc.find("str", {"name": "conveyanceText"}).text,
        "markDesc": descriptions
    }
    trademark_assignments.append(record)

len(trademark_assignments)

# %%
# 65.3 LOAD TRADEMARK ASSIGNMENTS INTO PANDAS
import pandas as pd
df = pd.DataFrame(trademark_assignments)
df[['conveyanceText', 'markDesc']]

# %%
# 65.4 ANALYSE THE RECORDS WITH GPT
from ipython_secrets import get_secret
from openai import OpenAI

KEY = get_secret("OPENAI_API_KEY")
openai = OpenAI(api_key=KEY)

def analyze_trademarks(df):
    results = []
    for index, row in df.iterrows():
        prompt = f"Analyze the following trademark information and explain what it means:\n{row.to_json()}"

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )

        analysis = response.choices[0].message.content
        results.append(analysis)

    return results

# %%
# 65.5 REVIEW THE RESULTS

from IPython.display import display, Markdown

results = analyze_trademarks(df.iloc[-1:])

for a in results:
    display(Markdown(a))


# %% [markdown]
# # 66. Seeding a gold standards database

# %%
# 66.1 SEEDING WITH PATENT CLIENT
import pandas as pd
from patent_client import PatentBiblio
# Search patent bibliographic records since ~1979
battery_patents = (
    PatentBiblio.objects.filter(patent_title="battery", assignee="tesla")
    .limit(5)
    .values(
        title="patent_title",
        code="main_classification_code",
        file_date="app_filing_date",
        appl_id="appl_id",
        pub_date="publication_date",
        pub_num="publication_number",
        pages="document_structure.page_count",
    )
)
df = battery_patents.to_pandas()
df

# %%
# 66.2 SEEDING WITH BIGQUERY
"""
! pip install google-cloud-bigquery db-dtypes google-cloud-bigquery-storage
"""

from google.cloud import bigquery
import os
HOME = os.path.expanduser('~')
APP_CREDS = f"{HOME}/.config/gcloud/gpt-cookbook-5462054842c5.json"
client = bigquery.Client.from_service_account_json(APP_CREDS)
sql = """
SELECT * 
FROM `bigquery-public-data.samples.shakespeare` 
ORDER BY `word_count` DESC
LIMIT 10
"""

df = client.query(sql).to_dataframe()

df

# %% [markdown]
# # 67. Topic classification

# %%
# 67.1 SETUP ENVIRONMENT
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

# %%
# 67.2 PREPARE DATA

# Load the dataset
dataset = load_dataset("batterydata/paper-abstracts")

# Preprocess the dataset
def preprocess_function(examples):
    # Tokenize the abstract and provide labels
    return tokenizer(
        examples["abstract"], truncation=True, padding="max_length", max_length=512
    )

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Ensure labels are integers (modify this according to your dataset)
tokenized_datasets = tokenized_datasets.map(
    lambda examples: {"label": [1 if l =='battery' else 0 for l in examples["label"]]},
    batched=True,
)

# Split the dataset
train_dataset = (
    tokenized_datasets["train"].shuffle(seed=42).select(range(5000))
)  # Adjust the range as needed
eval_dataset = (
    tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
)  # Adjust the range as needed


# %%
# 67.3 PREPARE CLASSIFIER
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


# %%
# 67.4 DEFINE THE TRAINING CONFIGURATION
def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": accuracy, "f1": f1}
    
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Define the trainer with compute_metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# %%
# 67.5 EVALUATE BASELINE PERFORMANCE

# base model evaluation
eval_result = trainer.evaluate()
print("BASELINE RESULTS")
eval_result

# %%
# 67.6 FINE-TUNING THE MODEL
trainer.train()

# %%
# 67.7 POST-FINE-TUNING EVALUATION
eval_result = trainer.evaluate()
print("FINE TUNED EVALUATION")
eval_result
