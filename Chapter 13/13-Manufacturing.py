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
# # 68. Streamlining quality control

# %%
# 68.1 DATASET SYNTHESIS
import pandas as pd
import numpy as np

# Synthesize dataset
np.random.seed(0)
data = {
    "Length": np.random.normal(10, 0.5, 1000),
    "Width": np.random.normal(5, 0.3, 1000),
    "Weight": np.random.normal(500, 50, 1000),
    "Color_Consistency": np.random.choice(["Consistent", "Inconsistent"], 1000),
    "Quality_Check": np.random.choice(["Pass", "Fail"], 1000, p=[0.95, 0.05]),
}

df = pd.DataFrame(data)

# %%
# 68.2 GPT INTEGRATION FOR ANOMALY DETECTION
from openai import OpenAI
from ipython_secrets import get_secret

# Initialize OpenAI
KEY = get_secret("OPENAI_API_KEY")
openai = OpenAI(api_key=KEY)

SYSTEM_PROMPT = """
You are a manufacturing QCinspector.
Your user will prompt you with a dataset.
Analyze the dataset for quality control insights.
Identify any anomalies
"""


def analyze_quality_control(data):
    chat_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": data},
    ]

    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo", messages=chat_history, max_tokens=100
    )

    return completion.choices[0].message.content


# Convert DataFrame to string and limit rows for the prompt
data_str = df.head(50).to_string(index=False)
analysis = analyze_quality_control(data_str)
print(analysis)


# %% [markdown]
# # 69. Programming manufacturing robots

# %%
"""
! pip install roboticstoolbox-python
"""

# %% [markdown]
# ## 69.1 System prompt

# %%
SYSTEM_PROMPT = """
You are an assistant helping me with a simulation of a Mico robot with 4 joints (RRRR). 
When I request a task, your role is to provide me with the necessary Python code to accomplish 
that task using the given object instance. YOu should also provide an explanation of what that code does. 
Only use the functions that have been predefined this prompt. 

The waist and wrist joints have unrestricted rotation. 
Their angle can be set from -360 to 360 degrees. 
The limits of the other joints are:

Shoulder: 47 to 313 degrees
Elbow: 19 to 341 degrees

Here is some sample code:

'''python
mb = MaxBot()
print(mr)

# Extend the arm fully horizontally
mb.set_joint_angles(0, 90, 180)

# Extend the arm fully vertically
mb.set_joint_angles(0, 180, 180)
'''

Use the commented lines in the above code as your fine tuning:

Other notes: 
- Do not use direct low-level motion commands, use only the functions defined for you.
- If in doubt, you should always ask for clarification. Never make assumptions.
- In terms of axis conventions, forward means positive X axis. Right means positive Y axis. Up means positive Z axis.

Here are the functions you can use to command the robot arm:

## rw.place_arm(self, waist, shoulder, elbow)
Sets angle, given in degrees, of the main joints

## rw.get_hand_orientation()
Returns the current orientation of the robot arm's end-effector.
"""

# %% [markdown]
# ## 69.2 Bridge GPT and the robotics toolbox
#
# See imports and variable assignments in `voltchat_robotics.py`

# %% [markdown]
# ## 69.3 MaxBot Class
#
# See class definition in `voltchat_robotics.py`

# %% [markdown]
# ## 69.4 Co-pilot
#
# See `def copilot()` in `voltchat_robotics.py`

# %% [markdown]
# ## 69.5 Initiate simulator

# %%
from voltchat_robotics import MaxBot, copilot
from swift import Swift
        
mb = MaxBot()

# Make and instance of the Swift simulator and open it
env = Swift()
env.launch(browser="notebook")

# Add the robot to the simulator
env.add(mb)

# %% [markdown]
# ## 69.6 Test the co-pilot

# %%
from IPython.display import display, Markdown

response = copilot("Extend the arm out fully pointing upwards at 45 degrees.")

display(Markdown(response))

# %%
# Code copied from copilot
mb.set_joint_angles(0, 45, 180)
env.step()

# %%
# Code expected from copilot
mb.set_joint_angles(0, 135, 180)
env.step()

# %% [markdown]
# # 70 Predictive Maintenance - Business analysis

# %% [markdown]
# There is no code in this recipe. See [NASA's PCOE repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) for the battery data.

# %% [markdown]
# # 71 Predicitive Maintenance - Data Engineering

# %%
# 71.1 PRE-FLIGHT CHECK
import requests

def get_final_download_url(url):
    response = requests.head(url, allow_redirects=True)
    return response.url


def get_file_info(url):
    response = requests.head(url)
    headers = response.headers

    file_info = {
        "File Size": headers.get("Content-Length"),
        "Content Type": headers.get("Content-Type"),
        "Last Modified": headers.get("Last-Modified"),
    }

    return response, file_info


initial_url = "https://data.nasa.gov/download/xg3n-ngei/application.zip"
final_url = get_final_download_url(initial_url)
print("Final downloadable URL:", final_url)
response, file_info = get_file_info(final_url)
print(response.status_code, file_info)

# %%
# 71.2 ZIP DOWNLOAD
import requests
from tqdm import tqdm

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

        with open(local_filename, "wb") as file:
            for data in r.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

download_file(final_url, 'data/battery_alt_dataset.zip')

# %%
# 71.3 FILE STRUCTURE ANALYSIS

import zipfile
import os

# Unzip the file
zip_file_path = "data/battery_alt_dataset.zip"
extract_to_path = "data/tmp"

with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(extract_to_path)


# Function to summarize the file tree
def summarize_file_tree(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            filepath = os.path.join(root, f)
            size = os.path.getsize(filepath)
            print(f"{subindent}{f} ({size} bytes)")


# Summarize the file tree for the extracted directory
summarize_file_tree(extract_to_path)

# %%
# 71.4 SUMMARISE THE CONTENTS

import os

def read_and_print_file_contents(file_path):
    try:
        with open(file_path, "r") as file:
            print(f"Contents of {file_path}:\n")
            print(file.read())
            print("\n" + "-" * 50 + "\n")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")


# Define the paths to the README files
base_path = "data/battery_alt_dataset"
readme_paths = [
    os.path.join(base_path, "README.txt"),
    os.path.join(base_path, "regular_alt_batteries", "README.txt"),
    os.path.join(base_path, "recommissioned_batteries", "README.txt"),
    os.path.join(base_path, "second_life_batteries", "README.txt"),
]

# Read and print the contents of each README file
for readme_path in readme_paths:
    read_and_print_file_contents(readme_path)

# %% [markdown]
# # 72 Preventative maintenance - Data science co-pilot

# %% [markdown]
# - No code in 72.1 - Observe
# - See code for 72.2 below
# - No code in 72.3 - Decide  
# - No code in 72.4 - Act

# %%
# 72.2 ORIENT
import os
import pandas as pd

def quick_look(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print(f"Analyzing: {file_path}")
                try:
                    df = pd.read_csv(file_path)

                    # Display basic information
                    print("Size of DataFrame:", df.size)
                    print("First few rows:")
                    print(df.head())
                    print("DataFrame Info:")
                    df.info()

                    # Check if start_date is monotonic increasing
                    if "start_date" in df.columns:
                        is_monotonic = df["start_date"].is_monotonic_increasing
                        print(f"Is 'start_date' monotonic increasing: {is_monotonic}")

                    # Get unique values in the start_date column
                    if "start_date" in df.columns:
                        unique_markers = df["start_date"].unique()
                        print(f"Unique markers in 'start_date': {unique_markers}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                print("\n" + "-" * 50 + "\n")


# Replace with the path to your dataset directory
dataset_directory = "data/battery_alt_dataset/"
quick_look(dataset_directory)

# %% [markdown]
# # Recipe 73 - Perspective

# %%
# 73.1 SPIN UP PERSPECTIVE
import pandas as pd
import panel as pn

pn.extension("perspective")
# Load the smallest CSV file
file_path = "data/battery_alt_dataset/regular_alt_batteries/battery20.csv"
df = pd.read_csv(file_path)

# Create a Perspective Pane
perspective_pane = pn.pane.Perspective(
    df, 
    height=600, 
    width=1000,
    title="Battery 20"
)

# Spin up a server which will show the Perspective Pane 
# in a new browser tab/window
bokeh = perspective_pane.show()


# %%
bokeh.stop()

# %% [markdown]
# - No code in 72.2 - Explore the data
# - No code in 72.3 - Discuss with Max Data
# - No code in 72.4 - Next steps with Max Data

# %% [markdown]
# # Recipe 74 Analysis with Scipy

# %%
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks

FEATURE = "temperature_battery"
NPEAKS = 5

# Load the data
df = pd.read_csv("data/battery20_SLICE.csv")  # Replace with your file path

# Identify peaks with temperature_battery > 100
peaks, _ = find_peaks(
    df[FEATURE], height=100, distance=df.shape[0] / (NPEAKS * 2)
)  # height parameter filters for high peaks
peak_times = df["time"].iloc[peaks]

# Calculating periodicity (average time difference between peaks)
periodicity = peak_times.diff().mean()

# Prepare the plot
fig = go.Figure()

# Plotting the temperature_battery
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df[FEATURE],
        mode="lines",
        name="Temperature Battery",
    )
)

# Plotting the peaks
fig.add_trace(
    go.Scatter(
        x=df["time"].iloc[peaks],
        y=df[FEATURE].iloc[peaks],
        mode="markers",
        name="Peaks",
        marker=dict(color="red", size=10),
    )
)

# Update layout
fig.update_layout(
    title="Temperature Battery with Peaks",
    xaxis_title="Time",
    yaxis_title="Temperature Battery",
)

# Show the plot
fig.show()

print(f"Average Periodicity: {periodicity}")

# %% [markdown]
# # Recipe 75 - Advanced visualisation with Plotly

# %%
# 75.1 IMPORT NECESSARY LIBRARIES AND DEFINE FIRST SUBPLOT
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def add_voltage_temperature_subplot(fig, df, row):
    fig.add_trace(
        go.Scatter(
            y=df["voltage_charger"],
            name="Charger Voltage",
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            y=df["temperature_battery"],
            name="Battery Temperature",
        ),
        row=row,
        col=1,
    )

# %%
# 75.2 CREATE A SUB-FUNCTION FOR THE SECOND SUBPLOT
def add_mode_subplot(fig, df, row):
    for mode_value, mode_name, color in zip(
        [-1, 0, 1], ["Discharge", "Rest", "Charge"], ["blue", "green", "red"]
    ):
        mode_indices = df["mode"] == mode_value
        fig.add_trace(
            go.Scatter(
                x=df.index[mode_indices],
                y=df["mode"][mode_indices],
                mode="markers",
                name=f"Mode {mode_name}",
                line=dict(color=color),
            ),
            row=row,
            col=1,
        )

# %%
# 75.3 CREATE A SUB-FUNCTION FOR THE THIRD SUBPLOT
def add_mission_type_subplot(fig, df, row):
    for mission_type_value, color in zip(
        [0, 1], ["cyan", "magenta"]
    ):
        mission_type_indices = df["mission_type"] == mission_type_value
        fig.add_trace(
            go.Scatter(
                x=df.index[mission_type_indices],
                y=df["mission_type"][mission_type_indices],
                mode="markers",
                name=f"Mission Type {mission_type_value}",
                line=dict(color=color),
            ),
            row=row,
            col=1,
        )

# %%
# 75.4 DEFINE THE LAYOUT CUSTOMIZATION FUNCTION
def update_layout(fig):
    # Update y-axes titles
    fig.update_yaxes(title_text="Voltage/Temperature", row=3, col=1)

    # Hide line, ticks, and grid on the first two rows
    hide_args = dict(
        showline=False, showgrid=False, zeroline=False, showticklabels=False, col=1
    )
    for i in [1, 2]:
        fig.update_xaxes(**hide_args, row=i)
        fig.update_yaxes(**hide_args, row=i)

    # Update layout
    fig.update_layout(
        title="Voltage Charger and Temperature Battery Analysis", showlegend=True
    )

# %%
# 75.5 DEFINE THE PIPELINE FUNCTION
def pretty_plotly(df, TIMESCALE):
    df = df[["time", "voltage_charger", "temperature_battery", "mode", "mission_type"]]

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.1, 0.1, 0.8],
        shared_xaxes=True,
        specs=[[{}], [{}], [{}]],
    )

    # Add subplots
    add_voltage_temperature_subplot(fig, df, row=3)
    add_mode_subplot(fig, df, row=1)
    add_mission_type_subplot(fig, df, row=2)

    # Update layout
    update_layout(fig)

    return fig

# %%
# 75.6 INVOKE THE PIPELINE FUNCTION
import pandas as pd
file_path = "data/battery_alt_dataset/regular_alt_batteries/battery20.csv"
df = pd.read_csv(file_path)

# Call the function with a sample of the data
fig = pretty_plotly(df[:25000], 60 * 60)
fig.show()
