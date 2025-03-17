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
# # Recipes from earlier chapters
#
# ```python
# '''
# CHAPTER 2
# '''
# # Hello NLP Toolbox
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# ... 
#
# '''
# CHAPTER 3
# '''
# # Tokenising with GPT & Hugging Face
# # Padding and Truncation in Practice with GPT 
# # Encoding in Practice with GPT	from transformers 
# from transformers import GPT2Tokenizer
# ...
#
# '''
# CHAPTER 5
# '''	
# # Simple Fine Tuning
# # Text generation fine tuning 
# # Q&A fine tuning	
#
# from transformers import (
#     GPT2LMHeadModel, 
#     GPT2Tokenizer, 
#     TextDataset, 
#     Trainer, 
#     TrainingArguments,
#     DataCollatorForLanguageModeling
# )
# ```

# %% [markdown]
# ## 32. Text Generation

# %%
'''
! pip install transformers
'''
# Import the necessary libraries to begin working with the model:
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load both the model and its corresponding tokenizer:
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define the text prompt
prompt_text = "Once upon a time in a land far away,"

# Tokenize the Prompt
inputs = tokenizer(
    prompt_text,
    return_tensors="pt",
    #padding=True,
    #truncation=True
)

output = model.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    pad_token_id=tokenizer.eos_token_id,
    max_length=100,
    temperature=0.7,
    top_p=0.9, 
    top_k=40, 
    repetition_penalty=1.2
)

# decode the generated tokens back into human-readable text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)


# %% [markdown]
# ## 33. Fine-tuning Models for Custom Tasks

# %%
# 33.1 Install required libraries
'''
!pip install evaluate
!pip install "transformers[torch]"
!pip install datasets
!pip install scikit-learn
!pip install accelerate -U
'''

# %%
# 33.2 Load and prepare the dataset

from datasets import load_dataset
import evaluate

# Load the accuracy metric
accuracy = evaluate.load("accuracy")

# Load the dataset
dataset = load_dataset('imdb', split={'train': 'train[:10]', 'test': 'test[:10]'})

type(dataset), dataset['train'], dataset['test']

# %%
# 33.3 Tokenize the dataset
from transformers import GPT2Tokenizer

def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        truncation=True, 
        padding='max_length', 
        max_length=128
    )

tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2-medium",
    clean_up_tokenization_spaces=True
    )
tokenizer.pad_token = tokenizer.eos_token

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# %%
# 33.4 Modify the dataset format to fit GPT-2's Sequence Classification task

COLS = [
    'input_ids', 
    'attention_mask', 
    'label'
]

small_train_dataset = tokenized_datasets["train"].with_format("torch")
small_train_dataset.set_format(
    type=small_train_dataset.format["type"], 
    columns=COLS
)

small_test_dataset = tokenized_datasets["test"].with_format("torch")
small_test_dataset.set_format(
    type=small_test_dataset.format["type"], 
    columns=COLS
)


# %%
# 33.5 Initialize the model for sequence classification
from transformers import GPT2ForSequenceClassification

# Initialize with 2 labels
model = GPT2ForSequenceClassification.from_pretrained(
    "gpt2-medium", 
    num_labels=2
)
model.config.pad_token_id = tokenizer.pad_token_id

# %%
# 33.6. Define custom functions
import torch

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    predictions = torch.argmax(logits, dim=-1)
    return accuracy.compute(
        predictions=predictions.numpy(), 
        references=labels
    )

# Define data_collator function
def custom_data_collator(data):
    input_ids = [item['input_ids'] for item in data]
    attention_masks = [item['attention_mask'] for item in data]
    labels = [item['label'] for item in data]
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.tensor(labels)
    }

# %%
# 33.7. Training arguments and trainer
from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_dir='./logs',
    learning_rate=2e-5,
    output_dir="./results",
    num_train_epochs=1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_test_dataset,
    tokenizer=tokenizer,
    data_collator=custom_data_collator,
    compute_metrics=compute_metrics
)



# %%
trainer.train() # Start the training process
results = trainer.evaluate() # Evaluate the model on the test dataset


# %%
for k in results:
    print(f'{results[k]:.2f}', k)

# %% [markdown]
# ## 34. Sentiment Analysis

# %%
'''
34.1 Requirements
! pip install xformers
'''
from transformers import pipeline

'''
34.2 CREATE A SENTIMENT ANALYSIS PIPELINE
'''
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_analyzer = pipeline(
    'sentiment-analysis',
    model=model_name,
    tokenizer=model_name
)

'''
34.3 PREPARE THE TEXT
'''
text_to_analyze = "I love using Hugging Face's Transformers library!"


# %%

'''
34.4 ANALYZE THE SENTIMENT
'''
result = sentiment_analyzer(text_to_analyze)


# %%

'''
34.5 INTERPRET AND DISPLAY THE RESULT
'''   
sentiment = result[0]['label']
confidence = result[0]['score']
print(f"The sentiment is {sentiment} with a confidence of {confidence}.")

# %% [markdown]
# ## 35. Question Answering

# %%
'''
35.1 SUPPRESS WARNINGS
'''
from transformers import pipeline
from transformers import logging
logging.set_verbosity_error()

'''
35.2 CREATE A QUESTION ANSWERING PIPELINE
'''
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
question_answering_model = pipeline('question-answering', model=model_name, tokenizer=model_name)

'''
35.3 PREPARE THE CONTEXT AND QUESTION
'''
context = "Hugging Face is creating a tool that democratizes AI."
question = "What is Hugging Face creating?"

'''
35.4 ASK THE QUESTION AND GET THE ANSWER
'''
result = question_answering_model({'context': context, 'question': question})

'''
35.5 INTERPRET AND DISPLAY THE RESULT
'''
answer = result['answer']
confidence = result['score']
print(f"The answer is: {answer} (confidence: {confidence})")

# %% [markdown]
# ## 36. Text Summarization

# %%
'''
36.1 Libraries
'''
from transformers import pipeline

'''
36.2 CREATE A SUMMARIZATION PIPELINE
'''   
model_name = "facebook/bart-large-cnn"
summarizer = pipeline('summarization', model=model_name, tokenizer=model_name)

'''
36.3 PREPARE THE TEXT TO BE SUMMARIZED
'''   
text_to_summarize = '''
The Hugging Face Transformers library is a cutting-edge collection of pre-trained models, 
tools, and resources designed to facilitate research and production in 
natural language processing (NLP). With its roots in the research community, 
the library has become a standard for academics and professionals working in machine learning. 
Offering a wide array of models such as BERT, GPT-2, GPT-3, and more, the library enables 
tasks ranging from text classification, sentiment analysis, and question answering to machine 
translation, text summarization, and beyond.

What sets the Transformers library apart is its user-friendly design, allowing both newcomers 
and seasoned experts to work with state-of-the-art models. Its architecture provides an 
abstraction layer over complex deep learning frameworks, ensuring that researchers and 
practitioners can focus on building applications without getting bogged down in implementation 
details.

Moreover, the Transformers library fosters a vibrant and collaborative community, 
encouraging users to contribute and share models, training scripts, and other resources. 
The Hugging Face Model Hub is an open platform that hosts thousands of pre-trained models, 
fine-tuned for specific tasks and languages, providing a seamless way to discover and 
load models with just a few lines of code.

Integration with popular deep learning frameworks like PyTorch and TensorFlow ensures that 
the library remains flexible and extensible. Continuous updates, contributions from the community, 
and adherence to best practices keep the library at the forefront of innovation in the field of AI.

The combination of accessibility, comprehensive documentation, a wide selection of models, 
collaboration opportunities, and continual development make the Hugging Face Transformers library a 
vital tool for anyone working in the rapidly evolving world of NLP. Its mission to democratize AI 
by making cutting-edge models available to all resonates with users around the globe, solidifying 
its status as a go-to resource for modern natural language processing."
'''

'''
36.4 GENERATE THE SUMMARY
'''
summary = summarizer(text_to_summarize, min_length=18, max_length=50)

'''
36.5 INTERPRET AND DISPLAY THE RESULT
The result will be a list containing a dictionary with the summary text:
'''
summary_text = summary[0]['summary_text']
print(f"The summarized text is: {summary_text}")

# %% [markdown]
# ## 37. Language Translation

# %%
'''
37.1 LIBRARIES
! pip install sentencepiece sacremoses
'''

from transformers import MarianMTModel, MarianTokenizer

'''
37.2 CHOOSE SOURCE AND TARGET LANGUAGES
'''
src_lang = 'en'
tgt_lang = 'de'
model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'

'''
37.3 LOAD TOKENIZER AND MODEL
'''
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

'''
37.4 PREPARE THE TEXT FOR TRANSLATION
'''
text_to_translate = "Hugging Face is democratizing AI."

'''
37.5 TOKENIZE AND TRANSLATE
'''
inputs = tokenizer(
    [text_to_translate], 
    return_tensors='pt', 
    truncation=True, 
    padding='max_length'
)
translated = model.generate(**inputs, max_new_tokens=50)
translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
print(f"Translated text: {translated_text[0]}")
