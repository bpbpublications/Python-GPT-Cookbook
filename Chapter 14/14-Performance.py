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
PROMPTS = [
    "Explain Newton's laws of motion.",
    "Write a short poem about the sea.",
    "Summarize the plot of 'Romeo and Juliet'.",
    "Generate a list of healthy breakfast options.",
    "Translate 'Hello, how are you?' into French.",
    "What are the health benefits of regular exercise?",
    "Create a short story about a lost puppy.",
    "Provide three tips for effective time management.",
    "What is the greenhouse effect?",
    "Suggest a recipe for a vegan pasta dish.",
    "Explain the basics of quantum computing.",
    "Write a motivational quote about perseverance.",
    "Describe the history of the internet.",
    "How does photosynthesis work?",
    "Give an overview of Aristotle's philosophy.",
    "Recommend five books on personal development.",
    "Explain the concept of machine learning.",
    "Write a joke about artificial intelligence.",
    "Summarize today's top news stories.",
    "Describe the steps in baking sourdough bread.",
]

# %%
from openai import OpenAI
from ipython_secrets import get_secret
from IPython.display import display, Markdown
import json

KEY = get_secret("OPENAI_API_KEY")
client = OpenAI(api_key=KEY)


# %%

my_prompt = PROMPTS[0]
response = client.completions.create(
    model="gpt-3.5-turbo-instruct", prompt=my_prompt, max_tokens=50
)

md = f"""
> {my_prompt}
```json
{json.dumps(response.dict(), indent=4)}
```
"""
display(Markdown(md))

# %% [markdown]
# # 76. API request parallel processor

# %%
# 76.1 ENVIRONMENT
'''
! pip install requests
'''

# %%
# 76.2 PARALLEL REQUEST FUNCTION
from openai import OpenAI
from ipython_secrets import get_secret

KEY = get_secret("OPENAI_API_KEY")
client = OpenAI(api_key=KEY)


def make_request(prompt):
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct", prompt=prompt, max_tokens=50
    )
    return response.choices[0].text

# %%
# 76.3 IMPLEMENTING PARALLEL PROCESSING
from concurrent.futures import ThreadPoolExecutor

def parallel_requests(prompt_list, max_workers=1):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(make_request, prompt_list))
    return results

# %%
# 76.4 PERFORMANCE COMPARISON
import time

# Sequential requests
start_time = time.time()
for data in PROMPTS[0:2]:
    make_request(data)
end_time = time.time()
print(f"Sequential Time: {end_time - start_time} seconds")

# Parallel requests
start_time = time.time()
parallel_requests(PROMPTS[0:2], max_workers=2)
end_time = time.time()
print(f"Parallel Time: {end_time - start_time} seconds")

# %% [markdown]
# # 77. Vertical Scaling

# %% [markdown]
# ## 77.1 Simple application
#
# See `vertical/app.py`

# %% [markdown]
# ## 77.2 Setting up your Docker environment
#
# See `vertical/dockerfile` and `vertical/requirements.txt`

# %% [markdown]
# ## 72.3 Building and running your Docker container

# %% [markdown]
# ```bash
# % cd vertical
# % docker build -t my-spacy-app .
# % docker run -d --name=AOK -p 4000:8000 my-spacy-app
# ```

# %% [markdown]
# ## 72.4 Scaling vertically
#
# ```bash
# % docker run -d --name=MEM -p 4001:8000 --memory=900m my-spacy-app
# % docker run -d --name=CPU -p 4002:8000 --cpus=1 my-spacy-app
# % docker run -d --name=MIN -p 4003:8000 --cpus=1 --memory=900m my-spacy-app
# % python test.py 3
# ```

# %% [markdown]
# # Horizontal Scaling

# %% [markdown]
# ## 78.1 Blueprinting your Docker environment
#
# See `horizontal/dockerfile` and `horizontal/requirements.txt`

# %% [markdown]
# ## 78.2 Sentiment analysis application
#
# See `horizontal/app.py`

# %% [markdown]
# ## 78.3 Building and running
#
# ```bash
# % cd horizontal
# % docker build -t my-torch-app .
# % docker run -d --name instance1 --cpus=1 -p 8001:8000 my-torch-app
# ```

# %% [markdown]
# ## 78.4 Performance testing
#
# ```bash
# % docker run -d --name instance2 --cpus=1 -p 8002:8000 my-torch-app
# % python test.py 6 1
# % python test.py 6 2
