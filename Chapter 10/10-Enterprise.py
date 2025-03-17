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
# # 55. Chat application

# %%
# 55.1 Initial set up and imports

import panel as pn
from openai import AsyncOpenAI

SYSTEM_KWARGS = dict(
    user="System",
    respond=False,
)

pn.extension()

# %%
# 55.2 Setting the API key
def set_client_key(key):
    if not key.startswith("sk-"):
        chat_interface.send("Please enter a valid OpenAI key!", **SYSTEM_KWARGS)
        return

    aclient.api_key = key
    chat_interface.disabled = False

    chat_interface.send(
        "Your OpenAI key has been set. Feel free to minimize the sidebar.",
        **SYSTEM_KWARGS,
    )


# %%
# 55.3 callback function
async def callback(
    contents: str,
    user: str,
    instance: pn.chat.ChatInterface,
):
    response = await aclient.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": contents}],
        stream=True
    )
    message = ""
    async for chunk in response:
        part = chunk.choices[0].delta.content
        if part is not None:
            message += part
            yield message

# %%
# 55.4 UI components and initialisation
key_input = pn.widgets.PasswordInput(placeholder="sk-...", name="OpenAI Key")
pn.bind(set_client_key, key=key_input, watch=True)
aclient = AsyncOpenAI(api_key="")
chat_interface = pn.chat.ChatInterface(callback=callback, disabled=True)
chat_interface.send(
    "First enter your OpenAI key in the sidebar, then send a message!", **SYSTEM_KWARGS
)
pn.template.MaterialTemplate(
    title="OpenAI ChatInterface with authentication",
    sidebar=[key_input],
    main=[chat_interface],
).show()

# %% [markdown]
# # 56 Indexing chats with Elasticsearch

# %%
# 56.1 Install and run ELK in Docker

# ! curl -fsSL https://elastic.co/start-local | sh

# %%
# 56.2 Implementation in Python using Elasticsearch DSL
from elasticsearch_dsl import Document, Text, Keyword, Date, connections

es = connections.create_connection(
    hosts=["http://localhost:9200"],
    basic_auth=('elastic', 'QHDGM0dr')
)


# %%

class Chat(Document):
    message = Text(analyzer="standard")
    user = Keyword()
    user_role = Keyword()
    timestamp = Date()
    conversation_id = Keyword()
    topic = Keyword()

    class Index:
        name = "voltchat"


Chat.init()

# %%
# 56.3 Indexing simulated conversations
import random
from datetime import datetime
from openai import OpenAI

# Define users and topics
users = [
    {"name": f"User{i}", "role": random.choice(["Engineer", "Manager", "Sales", "HR"])}
    for i in range(10)
]
topics = [
    "Battery Technology",
    "Market Trends",
    "Sustainability",
    "Product Development",
    "Customer Feedback",
    "Sales Strategy",
]

adjectives = [
    "terrible",
    "great",
    "poor",
    "awful",
    "fantastic",
    "the best",
]

# Simulate conversations
def simulate_conversations(num_messages, num_threads):
    for _ in range(num_threads):
        conversation_id = random.randint(1000, 9999)
        topic = random.choice(topics)
        for _ in range(num_messages):
            user = random.choice(users)
            adjective = random.choice(adjectives)
            # Generate message using GPT (mocked here)
            message = f"{topic} is {adjective}"
            # Index message in Elasticsearch
            chat = Chat(
                message=message,
                user=user["name"],
                user_role=user["role"],
                timestamp=datetime.now(),
                conversation_id=str(conversation_id),
                topic=topic,
            )
            chat.save()


simulate_conversations(50, 10)  # Example usage

# %%
# 56.4 Panel Dashboard for NLTK analysis
"""
! pip install hvplot
"""
import panel as pn
import pandas as pd
import hvplot.pandas
from elasticsearch_dsl import Search
from nltk.sentiment import SentimentIntensityAnalyzer

pn.extension()


# %%
# Fetch data from Elasticsearch
def fetch_chat_data():
    s = Search(index="voltchat").query("match_all")
    responses = s.scan()
    return pd.DataFrame([r.to_dict() for r in responses])


# Perform NLTK analysis
def analyze_sentiment(df):
    sia = SentimentIntensityAnalyzer()
    df["sentiment"] = df["message"].apply(lambda x: sia.polarity_scores(x)["compound"])
    return df


# Create Panel dashboard
def create_dashboard():
    df = fetch_chat_data()
    df_analyzed = analyze_sentiment(df)
    sentiment_plot = df_analyzed["sentiment"].hvplot.hist()
    return pn.Column(sentiment_plot)


dashboard = create_dashboard()
bokeh_server = dashboard.show()

# %%
bokeh_server.stop()

# %% [markdown]
# # 57. Summarize incoming email

# %%
# !pip install pydantic --force-reinstall
pip install fastapi --force-reinstall
# ! pip install bump-pydantic
# ! ./bump.sh

# %%
# 57.1 Install prefect
# ! pip install -U prefect
# ! prefect version

# %% [markdown]
# ### 57.2 Connect to a Prefect API
# Run this in a terminal, since it is a service. 
#
# `% prefect server start`
#
# It could run in a Notebook, but it would block the thread and stop us from doing anything else.
#
# Now you can open the Prefect dashboard in your browser at http://localhost:4200
#

# %% [markdown]
# ### 57.3 Write a Prefect flow
#

# %%
# 57.4 Run the flow
# ! python prefect_email.py

# %% [markdown]
# ### 57.4 Create a Process work pool:
#

# %%
# ! prefect work-pool create --type process my-work-pool

# %%
# ! prefect work-pool ls

# %% [markdown]
# Run this in a terminal, since it is a service. 
#
# `% prefect worker start --pool my-work-pool`
#
# If you run it in a Notebook it will block the thread and stop you from doing anything else.

# %%
# 57.5 Deploy and schedule flow

# ! python prefect_local_deploy.py

# %%
# Assuming existing VoltChat setup

async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    if "status of ETL process" in contents:
        response = "ETL process is currently running."  # Simulate status response
    elif "content of the processed report" in contents:
        response = fetch_report_summary()  # Function to fetch summary from Elastic
    else:
        response = await aclient.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": contents}],
            stream=True,
        )
    message = ""
    async for chunk in response:
        part = chunk.choices[0].delta.content if isinstance(chunk, dict) else chunk
        if part is not None:
            message += part
            yield message

def fetch_report_summary():
    # Elasticsearch query to fetch the latest report summary
    # Placeholder for actual Elasticsearch query
    return "Summary of the latest report: [Report details]"

# %% [markdown]
# # 58. DBA Co-pilot
#

# %%
# 58.1 Install PosgreSQL in Docker
# ! docker pull postgres
# ! docker run -itd --name some-postgres -e POSTGRES_PASSWORD=foobar -p 5432:5432 postgres 
# ! docker ps

# %%
# 58.2 Create database
import psycopg
CONN = "postgres:foobar@localhost:5432"


with psycopg.connect(f'postgresql://CONN') as conn:
    conn.autocommit = True
    conn.execute("CREATE DATABASE cookbook")

# %%
# 58.3 Create table

DDL = """
    CREATE TABLE grid_data (
        record_id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        meter_id VARCHAR(50),
        data_type VARCHAR(50),  -- e.g., 'half_hourly', 'daily', 'weekly', 'monthly'
        metric_name VARCHAR(100),  -- e.g., 'energy_consumption', 'peak_demand'
        value DECIMAL,
        additional_info JSONB  -- for any miscellaneous data points
    );
"""

with psycopg.connect(f"postgresql://{CONN}/cookbook") as conn:
    
    # Open a cursor to perform database operations
    with conn.cursor() as cur:

        # Execute a command: this creates a new table
        cur.execute(DDL)

        # Make the changes to the database persistent
        conn.commit()

# %%
# 58.4 Mock Data

MOCK = """
    -- Insert mock data using generate_series
    INSERT INTO grid_data (timestamp, meter_id, data_type, metric_name, value, additional_info)
    SELECT 
        generate_series(
            '2024-01-01 00:00:00'::timestamp,  -- Start time
            '2024-01-07 23:30:00'::timestamp,  -- End time
            '30 minutes'::interval            -- Interval
        ) AS timestamp,
        'meter_' || (random() * 10)::int,     -- Random meter_id (meter_0 to meter_9)
        CASE WHEN random() > 0.5 THEN 'half_hourly' ELSE 'daily' END,  -- Random data type
        CASE WHEN random() > 0.5 THEN 'energy_consumption' ELSE 'peak_demand' END,  -- Random metric
        round((random() * 1000)::numeric, 2), -- Random value between 0 and 1000, rounded to 2 decimal places
        jsonb_build_object('unit', 'kWh', 'source', 'simulation')  -- JSONB additional info
    FROM generate_series(
            '2024-01-01 00:00:00'::timestamp,
            '2024-01-07 23:30:00'::timestamp,
            '30 minutes'::interval
        );
"""
with psycopg.connect(f"postgresql://{CONN}/cookbook") as conn:
    with conn.cursor() as cur:
        cur.execute(MOCK)
        conn.commit()

# %%
with psycopg.connect(f"postgresql://{CONN}/cookbook") as conn:
    with conn.cursor() as cur:
        # Query the database and obtain data as Python objects.
        cur.execute("SELECT * FROM grid_data")
        res = cur.fetchone()
print(res)

# %%
# 58.5 Exploratory analysis with ChatGPT co-pilot
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# SQLAlchemy connection string
engine = create_engine(f"postgresql+psycopg://{CONN}/cookbook")

# Sample query to fetch and plot daily energy consumption
Q = """
    SELECT timestamp, value
    FROM grid_data
    WHERE data_type = 'daily' AND metric_name = 'energy_consumption'
    ORDER BY timestamp;
"""

# Fetch data into DataFrame
df = pd.read_sql(Q, engine)
# Convert timestamp to datetime if not already done
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Aggregate data by day, calculating the mean value
df_aggregated = df.resample('D', on='timestamp').mean()

# Plot the aggregated dataframe
df_aggregated.plot(y='value', legend=False)
plt.title('Daily Energy Consumption (Aggregated)')
plt.ylabel('Average Value')
plt.show()


