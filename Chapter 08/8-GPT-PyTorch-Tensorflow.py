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
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Define layers here

    def forward(self, x):
        # Define forward pass here
        return x

# %%
input_size = hidden_size = num_classes = 1

model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, num_classes)
)

# %%
learning_rate = 1

import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# %% [markdown]
# # 43. Implementing GPT with PyTorch

# %%
'''
!pip install torch
!pip install transformers
'''

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

model.eval()

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

attention_mask = torch.ones(input_ids.shape)

with torch.no_grad():
    output = model.generate(
        input_ids, 
        max_length=100, 
        num_return_sequences=5, 
        temperature=0.7, 
        attention_mask=attention_mask, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True 
    )

for i, sequence in enumerate(output):
    decoded_sequence = tokenizer.decode(sequence, skip_special_tokens=True)
    print(f"Generated Sequence {i + 1}: {decoded_sequence}")

# %% [markdown]
# # 44. Implementing GPT with TensorFlow

# %%
'''
!pip install tensorflow
!pip install transformers
'''
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = TFGPT2LMHeadModel.from_pretrained('gpt2-medium')

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="tf")

attention_mask = tf.ones(input_ids.shape, dtype=tf.int32)

output = model.generate(
    input_ids, 
    max_length=100, 
    num_return_sequences=5, 
    temperature=0.7, 
    attention_mask=attention_mask, 
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True  
)

for i, sequence in enumerate(output):
    decoded_sequence = tokenizer.decode(sequence.numpy(), skip_special_tokens=True)
    print(f"Generated Sequence {i + 1}: {decoded_sequence}")


# %% [markdown]
# # 45. Converting GPT Models: PyTorch to TensorFlow

# %%
# 45.1 LOAD AND SAVE MODEL WEIGHTS
from transformers import TFGPT2LMHeadModel, GPT2LMHeadModel

# Load your pre-trained PyTorch model
pytorch_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Save the PyTorch model weights
pytorch_model.save_pretrained('./pytorch_gpt2/')

# Load the TensorFlow model from the saved PyTorch weights
tensorflow_model = TFGPT2LMHeadModel.from_pretrained('./pytorch_gpt2/', from_pt=True)

# %%
# 45.2 COMPARE OUTPUT
import tensorflow as tf
import torch
import numpy as np

def calculate_mse(tensor1, tensor2):
    return ((tensor1 - tensor2) ** 2).mean()

input_ids = tf.constant([[1, 2, 3]])

tf_output = tensorflow_model(input_ids=input_ids)
tf_output = tf_output.logits.numpy()

pt_output = pytorch_model(input_ids=torch.tensor(input_ids.numpy()))
pt_output = pt_output.logits.detach().numpy() 

mse = calculate_mse(tf_output, pt_output)
print(f"Mean Squared Error between TensorFlow and PyTorch outputs: {mse}")

# %% [markdown]
# # 46. Converting GPT Models: TensorFlow to PyTorch

# %%
from transformers import GPT2LMHeadModel, TFGPT2LMHeadModel, GPT2Tokenizer

tensorflow_model = TFGPT2LMHeadModel.from_pretrained('gpt2-medium')

tensorflow_model.save_pretrained('./path_to_saved_model/')

pytorch_model = GPT2LMHeadModel.from_pretrained('./path_to_saved_model/', from_tf=True)
