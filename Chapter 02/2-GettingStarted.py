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
# # Recipes

# %% [markdown]
# ## 5. iPython Secrets

# %%
# 5.1
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

# %%
# 5.2
'''
! pip install ipython-secrets
'''

from ipython_secrets import get_secret
import openai
openai.api_key = get_secret('OPENAI_API_KEY')

# %% [markdown]
# ## 6. Hello NLP Toolbox

# %%
# 6.1
from ipython_secrets import get_secret
from openai import OpenAI # openai>=1.0.0

KEY = get_secret('OPENAI_API_KEY')
client = OpenAI(api_key=KEY)

prompt="Translate the following English text to French: 'Hello, World!'"

response = client.completions.create(
    model="gpt-3.5-turbo-instruct", 
    prompt=prompt, 
    max_tokens=60
)

print(response.choices[0].text)  # Output will be the translation

# %%
assert spacy.util.is_package("en_core_web_sm")

# %%
# 6.2
import spacy

try:
    assert spacy.util.is_package("en_core_web_sm")
    # Load English tokenizer, tagger, parser, NER, and word vectors
    nlp = spacy.load("en_core_web_sm")
    # Process the text
    text = "Hello, World!"
    doc = nlp(text)

    # Print tokens and part-of-speech tags
    for token in doc:
        print(token.text, token.pos_)
        
except Exception:
    print("After the system downloads the model, rerun the snippet")
    spacy.cli.download("en_core_web_sm")


# %%
# 6.3

from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt_text = "Hello, World!"
inputs = tokenizer.encode(prompt_text, return_tensors="pt")

outputs = model.generate(
    inputs, 
    max_length=50, 
    pad_token_id=tokenizer.eos_token_id,
    num_return_sequences=1, 
    no_repeat_ngram_size=2
)

for i, output in enumerate(outputs):
    print(f"Sample {i+1}: {tokenizer.decode(output, skip_special_tokens=True)}")


