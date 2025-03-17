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
# ## 38. Installing Annoy

# %%
# 38.1 Save Annoy DB to disk
'''
! pip install annoy
'''
from annoy import AnnoyIndex
import random

f = 40  

t = AnnoyIndex(f, 'angular')
for i in range(1000):
    v = [random.gauss(0, 1) for z in range(f)]
    t.add_item(i, v)

t.build(10) 
t.save('data/test.ann')

# %%
# 38.2 Load and get items from Annoy
u = AnnoyIndex(f, 'angular')
u.load('data/test.ann') 
print(u.get_nns_by_item(0, 10)) 

# %% [markdown]
# ## 39. Generating & Indexing embeddings of book descriptions

# %%
# 39.1 - Initialise
'''
! pip install openai annoy datasets tqdm ipywidgets
'''
from annoy import AnnoyIndex
import openai
from ipython_secrets import get_secret
openai.api_key = get_secret('OPENAI_API_KEY')

MODEL = 'text-embedding-ada-002'
BATCH_SIZE = 100

# %%
# 39.2 - Get the corpus
import datasets

# Download the data file, use the `train` portion
dataset = datasets.load_dataset(
    'Skelebor/book_titles_and_descriptions_en_clean', 
    split=f'train[0:{BATCH_SIZE}]'
)

len(dataset)

# %%
# 39.3 - Create the embeddings
response = openai.embeddings.create(
    input=dataset['description'],
    model=MODEL
)

print(response.usage)

# %%
# 39.4 - Build the index
DIMENSION = len(response.data[0].embedding)
t = AnnoyIndex(DIMENSION, 'angular')
t.on_disk_build('data/books.ann')

# Embed and insert
for i in range(0, BATCH_SIZE):
    txt = dataset[i]['description']
    v = response.data[i].embedding
    t.add_item(i, v)

# Build with 10 trees
print(t.build(10), DIMENSION)

# %%
# 39.5 - Perform a query
u = AnnoyIndex(DIMENSION, 'angular')
u.load('data/books.ann')
q = 'Book about an owl'
n = 5
res = response = openai.embeddings.create(
    input=q,
    model=MODEL
)

v = res.data[0].embedding
raw_search_results = u.get_nns_by_vector(v, n, include_distances=True)
top_hit = raw_search_results[0][0]
title = dataset[top_hit]['title']
desc = dataset[top_hit]['description']

print(res.usage)
print(title,'\n',desc[0:75], '...')
print(raw_search_results[1][0])

# %% [markdown]
# ## 40. Analysing book descriptions

# %%
# 40.1 Import library code
'''
! pip install matplotlib
! pip install seaborn
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import datasets
from sklearn.manifold import TSNE
from pandas.plotting import parallel_coordinates
from annoy import AnnoyIndex

# %%
# 40.2 Load the Annoy database
DIM = 1536  # See recipe 39
u = AnnoyIndex(DIM, 'angular')
u.load('data/books.ann')

# %%
# 40.3 Create a distance matrix
BATCH_SIZE = 100
dist_matrix = np.ndarray(shape=(BATCH_SIZE,BATCH_SIZE))
features = pd.DataFrame(columns=range(DIM))

for i in range(BATCH_SIZE):
    features.loc[i] = u.get_item_vector(i)
    for j in range(BATCH_SIZE):
        dist_matrix[i,j] = u.get_distance(i,j)

dist_df = pd.DataFrame(dist_matrix)

dist_df.shape

# %%
# 40.4 Plot a histogram
mean_dist = dist_df.mask(dist_df==0).mean()
mean_dist.hist()

# %%
# 40.5 Find the outliers
mean_dist[mean_dist > 0.75]

# %%
# 40.6 Print out the titles of the outliers
dataset = datasets.load_dataset(
    'Skelebor/book_titles_and_descriptions_en_clean', 
    split=f'train[0:{BATCH_SIZE}]'
)
for i in mean_dist[mean_dist > 0.75].index:
    t = dataset[i]['title']
    print(i,t)

# %% [markdown]
# ## 41. Answering questions on current Nobel Prize winners

# %%
# 41.1 Demonstrate knowledge cut-off
from IPython.display import display, Markdown
from ipython_secrets import get_secret
import openai

openai.api_key = get_secret('OPENAI_API_KEY')
EMBEDDING_MODEL = 'text-embedding-ada-002'
GPT_MODEL = "gpt-3.5-turbo"
# Send a request to the OpenAI API (> v1.0.0)
response = openai.chat.completions.create(
  model=GPT_MODEL,
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Which previous Nobel Prize for Physics is most similar to the 2024 winner?"},
    ]
)
answer = response.choices[0].message.content
display(Markdown(f'> {answer}'))

# %%
# 41.2 Download a corpus of current events

import requests
import io
import pandas as pd

def get_nobel(cat=None):

    # Base URL for the API
    url = 'http://api.nobelprize.org/2.1/nobelPrizes'
    params = {
        "nobelPrizeYear": 1900,
        "yearTo": 2024,
        "limit": 10000,
        "format": 'csv',
    }

    if cat:
        params["nobelPrizeCategory"] = cat

    # Send the request to the API
    response = requests.get(url, params=params)

    response.status_code
    df = pd.read_csv(
        io.StringIO(response.text),
        index_col=False
    )

    return df

df = get_nobel(cat="phy")
df.head()[['year', 'name']]

# %%
# 41.3 Tokenize
"""
!pip install nltk
"""
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import download as nltk_download

nltk_download('punkt_tab')

def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    tokens = [
        lemmatizer.lemmatize(t) # lemmatizes the words
        for t in word_tokenize(text.lower()) # creates tokens
        if t not in stopwords.words('english')
        and t.isalpha() # removes numbers
    ]

    return tokens

# %%
df["keywords"] = df.motivation.apply(tokenize)
df.iloc[0].keywords

# %%
# 41.4 Vectorize

from annoy import AnnoyIndex
import openai
from ipython_secrets import get_secret
openai.api_key = get_secret('OPENAI_API_KEY')
    
response = openai.embeddings.create(
    input=list(df.keywords.str.join(' ')),
    model='text-embedding-ada-002'
)

dim = len(response.data[0].embedding)
u = AnnoyIndex(dim, 'angular')

for i,v in enumerate(response.data):
    u.add_item(i, v.embedding)

u.build(10) # 10 trees

# %%
# 41.5 Get test data
# returns the 10 closest items, the first one is the item itself
for i,row in df.query('year == 2024').iterrows():
    nn = u.get_nns_by_item(i, 10) 
    break
df.iloc[nn][['name','year','category']]

# %%
# 41.6 Ask a question with context given
from IPython.display import display, Markdown

df['sentence'] = df.name + " " + df.category + " " + df.year.astype(str) + " " + df.keywords.str.join(' ')
context = "\n".join(df.query("year > 2000").sentence)

# Send a request to the OpenAI API (> v1.0.0)
response = openai.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": f"Answer questions based only on this context: {context}"},
        {"role": "user", "content": "Which previous Nobel Prize for Physics is most similar to the 2024 winner?"},
    ]
)
answer = response.choices[0].message.content
display(Markdown(f'> {answer}'))

# %% [markdown]
# ## 42. Exploring Nobel Prize motivations

# %%
# 42.1 Create bag of words
from collections import Counter

m = get_nobel('phy').motivation
sentence = " ".join(m.str.lower())
tokens = tokenize(sentence)

bag = pd.DataFrame(
    Counter(tokens).items(),
    columns=['word','frequency']
)

bag = bag.sort_values(by='frequency',ascending=False)

# %%
# 42.2 Show a simple word cloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud, get_single_color_func
wc = WordCloud(
        width = 800,
        height = 800,
        background_color ='white',
        min_font_size = 10
)

w = bag.set_index('word').frequency.to_dict()       
wc.fit_words(w)
grey = get_single_color_func('grey')
wc.recolor(color_func=grey)

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wc)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()

# %%
# 42.3 Analyse bag of words
import nltk
nltk.download('averaged_perceptron_tagger_eng')

bag['len'] = bag.word.str.len()
bag['pos'] = [t[1] for t in nltk.pos_tag(bag.word)]
bag.set_index('word').head(10).plot.barh(legend=False)

# %%
# 42.4 Simple scatter plot
import plotly.express as px

fig = px.scatter(
    bag, 
    y='len',
    x='frequency',
    text="word",
    color="pos",
    symbol="pos"
)
fig.update_traces(textposition='top center')
fig.update_layout(title_text='Nobel bag of words', title_x=0.5)
fig.show()

# %%
# 42.5 Embed and cluster
from annoy import AnnoyIndex
import numpy as np
import openai
from ipython_secrets import get_secret

def cluster(vec, metric):
    dim = len(vec[0])
    u = AnnoyIndex(dim, metric)

    for i,v in enumerate(vec):
        u.add_item(i, v)

    u.build(10) # 10 trees

    n = len(vec)
    labels = np.full(n, -1)  # Initialize all labels as -1

    for i in range(n):
        if labels[i] == -1:  # If the point is not yet labeled
            # Get nearest neighbour
            nn = u.get_nns_by_item(i, 2)
            labels[i] = nn[1]
    
    return labels

openai.api_key = get_secret('OPENAI_API_KEY')
    
response = openai.embeddings.create(
    input=list(bag.word),
    model='text-embedding-ada-002'
)

vec = [v.embedding for v in response.data]

metric = ["angular", "euclidean", "manhattan", "dot", "hamming"]
for m in metric:
    bag[m] = cluster(vec, m)

# %%
bag[metric].var()

# %%
bag[metric[0:3]].diff(periods=1, axis=1).sum()

# %%
(bag.angular - bag.manhattan).to_numpy().nonzero()


# %%
# 42.6 Levenshtein distance
import matplotlib.pyplot as plt
import nltk

n = len(bag)
lev = np.zeros(shape=(n,n))
for i in bag.index:
    for j in bag.index:
        lev[i,j] = nltk.edit_distance(bag.loc[i].word, bag.loc[j].word)
lev = pd.DataFrame(lev)
lev.mask(lev==0).min().hist()

bag['levenshtein'] =  lev.apply(lambda x: x.mask(x == 0).idxmin())

plt.imshow(lev, cmap='gray_r')
plt.show()

# %%
# 42.7 Fancy scatter plot

import plotly.express as px
bag["tshirt"] = bag.len.apply(lambda x: 1 if x < 7 else 2)
bag["pop"] = bag.frequency.apply(lambda x: 1 if x < 10 else 2) 

fig = px.scatter(
    bag.query("len > 2 and frequency > 2 and frequency < 50"), 
    y='angular',
    x='levenshtein',
    text='word',
    symbol="pos",
    size="frequency",
    height=1200,
    width=1200,
    color="pos",
    facet_row="tshirt",
    facet_col="pop"
)
fig.update_traces(textposition='top center')
fig.update_layout(title_text='Nobel bag of words ++', title_x=0.5)
fig.show()


