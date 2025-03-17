# %%
# 9. Byte-Pair Encoding principles
import re, collections

# Function to count the pairs of symbols in the vocabulary
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

# Function to performs the merge operation on the most frequent pair.
def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

# Sample text corpus
corpus = "low lower newest widest"
vocab = {' '.join(word): freq for word, freq in collections.Counter(corpus.split()).items()}

# Number of merge operations
num_merges = 5

for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)

print(vocab)


# %%
''' 
10. ENCODING AND DECODING WITH SENTENCEPIECE
! pip install sentencepiece
'''

import urllib.request
import io
import sentencepiece as spm

# Loads model from URL as iterator and stores the model to BytesIO.
model = io.BytesIO()
with urllib.request.urlopen(
    'https://raw.githubusercontent.com/google/sentencepiece/master/data/botchan.txt'
) as response:
  spm.SentencePieceTrainer.train(
      sentence_iterator=response, 
      model_writer=model, 
      vocab_size=1000,
      model_type="bpe")

# Directly load the model from serialized model.
sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
tokens = sp.encode('low lower newest widest')

for t in tokens:
    print(t, sp.decode(t))



# %%
'''
11. Tokenizing with GPT and Hugging Face
! pip install transformers
'''
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "Your sample text here !"
text = text.lower()  # Lowercasing
tokens = tokenizer.tokenize(text)  # Tokenization
input_ids = tokenizer.convert_tokens_to_ids(tokens)  # Converting to numerical input

tokens, input_ids

# %%
len(tokens[1])

# %%
'''
12. REMOVING STOP WORDS WITH NLTK
! pip install nltk
'''

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
text = "This is an example of removing stop words."
text = ' '.join([word for word in text.split() if word.lower() not in stop_words])

print(text)


# %%
# 13. String translation with Python
import string

text = "Hello, World! This is an example."

# Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

print(text)

'''
Hello World This is an example
'''

# %%
'''
14. STEMMING AND LEMMATIZATION WITH NLTK
! pip install nltk
'''

import nltk
nltk.download('wordnet')
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed = []
nouns = []
verbs = []
adjectives = []
words = ["running", "goodness", "better"]

for w in words:
    stemmed.append(stemmer.stem(w))
    nouns.append(lemmatizer.lemmatize(w))
    verbs.append(lemmatizer.lemmatize(w, pos="v"))
    adjectives.append(lemmatizer.lemmatize(w, pos="a"))
    
print(stemmed) 
print(nouns)
print(verbs)
print(adjectives)

# %%
WordNetLemmatizer().lemmatize('better', pos="n")


# %%
# 15. Standard library padding and truncation
def pad_or_truncate_sequences(sequences, max_length, padding_token=0):
    """
    Pads or truncates sequences to the given max_length.

    :param sequences: List of sequences (lists of integers or tokens)
    :param max_length: Desired length of the sequences
    :param padding_token: Token to use for padding (default is 0)
    :return: List of padded/truncated sequences
    """
    processed_sequences = []
    for sequence in sequences:
        # Truncate the sequence if it's longer than max_length
        truncated_sequence = sequence[:max_length]
        
        # Pad the sequence with padding_token if it's shorter than max_length
        padded_sequence = truncated_sequence + [padding_token] * (max_length - len(truncated_sequence))
        
        processed_sequences.append(padded_sequence)
        
    return processed_sequences

# Example usage
sequences = [
    [1, 2, 3, 4],
    [5, 6],
    [7, 8, 9, 10, 11]
]
max_length = 3
processed_sequences = pad_or_truncate_sequences(sequences, max_length)

for sequence in processed_sequences:
    print(sequence)

# %%
'''
16. PADDING AND TRUNCATION IN PRACTICE WITH GPT
! pip install transformers
'''
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

text = ["I love programming with GPT models.", "GPT is amazing!"]

# Tokenize with padding and truncation
tokens = tokenizer(
    text, 
    padding=True, 
    truncation=True, 
    max_length=50, 
    return_tensors="pt"
)

# %%
print(tokens)


# %%
# 17. Encoding in practice with GPT
from transformers import GPT2Tokenizer

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Sample text to be encoded
text = "I love programming with GPT models."

# Encoding the text
encoded_input = tokenizer(text, return_tensors="pt")

print(encoded_input)


# %%
# 18. How to count tokens with tiktoken
! pip install tiktoken

# %%
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    ''' Returns the number of tokens in a text string.'''
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

num_tokens_from_string("tiktoken is great!", "cl100k_base")



# %%
! pip install transformers

# %%
# 19. Imputing missing words with GPT
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the pad_token for GPT-2
tokenizer.pad_token = tokenizer.eos_token

# Well-known phrase with a missing word: "An apple a day keeps the doctor ____."
text = "An apple a day keeps the doctor"

# Encode text to tensor format
input_ids = tokenizer.encode(text, return_tensors="pt")

# Generate attention mask
attention_mask = input_ids.ne(tokenizer.pad_token_id).type(input_ids.dtype)

# Generate text using the model with specified parameters
output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=5,  # Limiting to 5 new tokens
    pad_token_id=tokenizer.eos_token_id,
)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the result
print(generated_text)