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
'''
20. Loading Pre-trained Word Embeddings
!pip install gensim
'''
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

nltk.download('punkt')

# Example small corpus
corpus = "This is a small corpus of text. It's only an example."

# Tokenizing
sentences = [word_tokenize(sentence) for sentence in sent_tokenize(corpus)]

# Training the model
model = Word2Vec(sentences, window=5, min_count=1, workers=4)

# Saving the model in binary format
model.wv.save_word2vec_format('data/word2vec.bin', binary=True)


# %%
from gensim.models import KeyedVectors

# Load Word2Vec embeddings
word2vec_path = 'data/word2vec.bin'
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

# Example: Using the loaded embeddings
vector = word2vec_model['example']  # Get the vector for the word 'example'
type(vector), len(vector)

# %% [markdown]
# NB the DIY word2vec.bin file created is suitable for a simple demonstration. To do NLP proper, install a suitable pre-trained Word2Vec model. For example, [Google has released a pre-trained Word2Vec model that has been trained on Google News data](https://code.google.com/archive/p/word2vec/). This containins 3 million words and phrases. The file is around 1.5 GB and is in binary format, making it suitable for use in the example above.
#
# Gensim provides a way to download several pre-trained models directly through their API. The example given is 1662.8MB which will take a while to download the first time it is used. For future loads, the local version (stored in the given location) is used.

# %%
import gensim.downloader as api
print(api.BASE_DIR)
word2vec_model = api.load("word2vec-google-news-300")
vector = word2vec_model['computer']
type(vector), len(vector)

# %%
# 21. Text Preprocessing for Embeddings
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

text = "This is an example sentence. Let's tokenize, clean, and normalize it!"

# Tokenizing
tokens = word_tokenize(text)

# Removing punctuation and making lowercase
tokens = [word.lower() for word in tokens if word.isalpha()]

# Removing stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

# Lemmatizing
lemmatizer = WordNetLemmatizer()
normalized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

print(normalized_tokens)  # Output may look like ['example', 'sentence', 'let', 'tokenize', 'clean', 'normalize']

# %%
'''
! pip install spacy
! python -m spacy download en_core_web_sm
'''
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Load the English language model
nlp = spacy.load('en_core_web_sm')

text = "This is an example sentence. Let's tokenize, clean, and normalize it!"

# Processing the text
doc = nlp(text)

# Tokenizing, removing punctuation, stopwords, and lemmatizing
normalized_tokens = [token.lemma_.lower() for token in doc if token.is_alpha and token.text.lower() not in STOP_WORDS]

print(normalized_tokens)  # Output may look like ['example', 'sentence', 'let', 'tokenize', 'clean', 'normalize']


# %%
# 22. Using OpenAI's Models with Text Input
import openai

# Set your API key
from ipython_secrets import get_secret
openai.api_key = get_secret('OPENAI_API_KEY')

# Define the text prompt
prompt_text = "Translate the following English text to French: 'Hello, World!'"

# Send a request to the OpenAI API (> v1.0.0)
response = openai.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt_text,
    max_tokens=60,
    temperature=0.7
)

# Extract the generated text from the response
generated_text = response.choices[0].text

print("Input Prompt:", prompt_text)
print("Generated Translation:", generated_text)

# %%
# 23. Calculating Similarity Between Embeddings
import numpy as np

# Define two word vectors
vector1 = np.array([2, 3, 4, 5])
vector2 = np.array([4, 3, 2, 1])

# Compute cosine similarity
cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

# Compute Euclidean distance
euclidean_distance = np.linalg.norm(vector1 - vector2)

print("Cosine Similarity:", cosine_similarity)
print("Euclidean Distance:", euclidean_distance)


# %%
from scipy.spatial.distance import cosine, euclidean
import numpy as np

# Define two word vectors
vector1 = np.array([2, 3, 4, 5])
vector2 = np.array([4, 3, 2, 1])

# Compute cosine similarity (note that scipy's cosine function returns dissimilarity, so we need to subtract from 1)
cosine_similarity = 1 - cosine(vector1, vector2)

# Compute Euclidean distance
euclidean_distance = euclidean(vector1, vector2)

print("Cosine Similarity:", cosine_similarity)
print("Euclidean Distance:", euclidean_distance)


# %%
# 24 Visualizing Embeddings Using t-SNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Sample word vectors (replace this with your actual word vectors)
word_vectors = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
words = ["word1", "word2", "word3", "word4"]

# Reduce dimensionality to 2D
tsne = TSNE(n_components=2, perplexity=len(word_vectors)-1)
word_vectors_2d = tsne.fit_transform(word_vectors)

# Plot the 2D representation
plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])

for i, word in enumerate(words):
    plt.annotate(word, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]))

plt.show()

# %%
from mpl_toolkits.mplot3d import Axes3D

# Reduce dimensionality to 3D
tsne = TSNE(n_components=3, perplexity=len(word_vectors)-1)
word_vectors_3d = tsne.fit_transform(word_vectors)

# Plot the 3D representation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(word_vectors_3d[:, 0], word_vectors_3d[:, 1], word_vectors_3d[:, 2])

for i, word in enumerate(words):
    ax.text(word_vectors_3d[i, 0], word_vectors_3d[i, 1], word_vectors_3d[i, 2], word)

plt.show()



# %%
"""
25. APPLYING EMBEDDINGS FOR TEXT CLASSIFICATION

25.1: Pre-processing
"""

import spacy

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text
def preprocess(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return tokens

text = "I love OpenAI's models. They are very powerful!"
cleaned_text = preprocess(text)
print("Cleaned Tokens:", cleaned_text)


# %%
# # ! pip install kagglehub
"""
25.2: Transforming Text into Embeddings
"""
import kagglehub

# Download latest version
path_to_glove = kagglehub.dataset_download("danielwillgeorge/glove6b100dtxt")

# %%
import numpy as np

# Function to load GloVe vectors
def load_glove_embeddings(glove_file):
    embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Load GloVe embeddings
dim = 100
glove_file = f'{path_to_glove}/glove.6B.{dim}d.txt'
embedding_model = load_glove_embeddings(glove_file)

# Example usage: get the embedding for a word
print(embedding_model.get('love', 'Word not found'))

# %%
import kagglehub
import pandas as pd
import numpy as np

# Function to transform text into embeddings
def text_to_embedding(tokens, model, dim):
    embeddings = np.array([model[word] for word in tokens if word in model])
    if embeddings.size == 0:
        return np.zeros(dim)
    return np.mean(embeddings, axis=0)

# Download the raw data
path_to_imdb = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

# Read a small sample of data
df = pd.read_csv(f'{path_to_imdb}/IMDB Dataset.csv', nrows=100)

# Clean the text
df['cleaned_text'] = df['review'].apply(preprocess)

# Apply the embedding transformation
df['embedding'] = df['cleaned_text'].apply(lambda x: text_to_embedding(x, embedding_model, dim))


# %%
"""
25.3: Building a Classification Model
"""

from sklearn.model_selection import train_test_split

# Prepare features (X) and labels (y)
X = np.stack(df['embedding'].values)
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values  # Convert 'positive' to 1, 'negative' to 0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Evaluate the model
accuracy = classifier.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")


# %%
"""
25.4: Postprocessing and Interpreting Results
"""

TEST_TEXT = [
    "OpenAI's models are very powerful!",
    "Kaggle is supercalifragilisticexpialidocious!",
    "I hated this product, it stinks :-(",
    "I love this product, it is amazing :-)"
]

for t in TEST_TEXT:
    
    cleaned_text = preprocess(t)
    new_embedding = text_to_embedding(cleaned_text, embedding_model, dim)

    # Predict the sentiment using the classifier
    predicted_class = classifier.predict([new_embedding])
    print("+" if predicted_class[0] == 1 else "-", t)



# %%
# 26. Handling Out-of-Vocabulary Words

def get_embedding_ignore_unknown(word, model):
    # Ignoring the Unknown Word
    return model.wv[word] if word in model.wv else None

def get_embedding_unknown_token(word, model, unknown_token="UNK"):
    # Using a Special "Unknown" Token
    return model.wv[word] if word in model.wv else model.wv[unknown_token]

# %%
from gensim.models import FastText

def get_embedding_average_neighbors(word, model):
    # Averaging Neighbors (Using Character Embeddings)
    if word in model.wv:
        return model.wv[word]
    else:
        ngrams = [word[i:i+3] for i in range(len(word) - 2)] # Creating 3-grams
        ngram_embeddings = [model.wv[ng] for ng in ngrams if ng in model.wv]
        return np.mean(ngram_embeddings, axis=0) if ngram_embeddings else None


# %%
from gensim.models import Word2Vec

def train_custom_model(sentences):
    # Training a Custom Model
    return Word2Vec(sentences, min_count=1)


# %%
import numpy as np

def get_embedding_random(word, model, embedding_size=100):
    # Random Initialization
    return model.wv[word] if word in model.wv else np.random.rand(embedding_size)


# %%
# Load pre-trained GloVe embeddings
# Example 1: Using pretrained embeddings
glove_model = api.load("glove-wiki-gigaword-100")
type(glove_model)

# %%
# Example sentence with OOV words
tweet = "OpenAI models are supercalifragilisticexpialidocious!"

# Handle the OOV words

for word in tweet.lower().split():

    if word in glove_model:
        vec = glove_model[word]
        print(word, len(vec))
    else:
        print(word, 'UNK')


# %%
# Example 2: Using the 20 Newsgroups Dataset for Text Classification
from sklearn.datasets import fetch_20newsgroups

# Fetch the dataset
newsgroups_data = fetch_20newsgroups(subset='train').data

# Preprocess text by tokenizing into sentences and words
sentences = [doc.lower().split() for doc in newsgroups_data]

# Train a custom Word2Vec model on the dataset
model = Word2Vec(sentences, vector_size=100, min_count=1)


# %%

print(len(newsgroups_data), len(sentences), model.corpus_count)

# %%

# %%
word = "neuralnetwork"

vec = get_embedding_ignore_unknown(word, model)
print("Ignore OOV:", vec)

vec = get_embedding_random(word, model)
print("Random OOV embedding:", len(vec))