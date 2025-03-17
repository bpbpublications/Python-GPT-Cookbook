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
# # 59 Generating sales emails and press releases

# %%
# 59.1 Mock up data
import yaml

MOCK_ENQUIRY = """
Subject: Inquiry About Energy Storage Solutions for Versatile Use

Dear 10X Batteries

I am seeking an energy storage solution that is reliable, efficient, and versatile enough to meet a variety of needs. My primary requirements include portability, long-term durability, and compatibility with renewable energy systems like solar panels. Additionally, I need a solution that performs well in diverse conditions, including outdoor and remote locations.

Here are a few scenarios I am looking to address:
	1.	Backup Power: I require a reliable option to keep essential appliances and devices running during power outages.
	2.	Off-Grid Energy Storage: A solution that integrates seamlessly with solar panels for a remote cabin and can support sustained energy needs.
	3.	Mobile Applications: Portability is important, as I plan to use the system for outdoor events, camping, and other recreational activities.
	4.	Efficiency: Fast charging and energy efficiency are key considerations to ensure minimal downtime.

I’m also interested in understanding the safety features and maintenance requirements of your solutions, as well as their compatibility with other renewable energy systems.

If you have a solution that aligns with these needs, I’d love to hear more about it and any recommendations you may have. Thank you in advance for your assistance, and I look forward to your reply.

Best regards

Max Mustermann
"""

MOCK_PRODUCT = """
Product Name: EcoBattery Pro 3000
Price (€): 2999
Features:
  Weight (kg): 12.5
  Capacity (kWh): 3.0
  C Value: 1.5
  Cycle Life (cycles): 5000
  Operating Temperature Range (°C): -20 to 60
  Charging Time (hours): 2
  Output Voltage (V): 48
  Efficiency (%): 95
  Dimensions (cm): 35 x 25 x 15
Customer Feedback: 
    - I love how efficient this product is, but I wish it came in more colors.
    - The battery life is incredible, but the charging time could be shorter.
    - It's lightweight and easy to carry, perfect for outdoor use.
    - The price is a bit high, but the performance is worth it.
    - I appreciate the wide operating temperature range; it’s great for extreme conditions.
    - The design is sleek, but it would be even better with more customization options.
    - The efficiency is impressive, but I’d like to see more detailed documentation.
    - It’s a reliable product for my off-grid setup, and the cycle life is unmatched.
"""

# Convert the YAML string to a Python dictionary
MOCK_PRODUCT = yaml.safe_load(MOCK_PRODUCT)


# %%
# 59.2 Personalised Email
MOCK_COMBO = f"""
{MOCK_PRODUCT}
---
{MOCK_ENQUIRY}
"""
prompt = f"Create a reply to the following enquiry and product data combo: /n {MOCK_COMBO}"

response = openai.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt, 
    max_tokens=150
)

print(response.choices[0].text.strip())

# %%
# 59.3 Sentiment analysis
import openai
from ipython_secrets import get_secret

# Set your API key
openai.api_key = get_secret('OPENAI_API_KEY')

def analyze_sentiment(feedback):
    
    # Send a request to the OpenAI API (> v1.0.0)
    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Label the sentiment of: '{feedback}' as POSTIVE, NEGATIVE or NEUTRAL",
        max_tokens=100
    )

    # Extract the sentiment
    return response.choices[0].text.strip().upper()

for s in MOCK_PRODUCT['Customer Feedback']:
    print(analyze_sentiment(s), ' => ', s)

# %%
# 59.4 Press release
prompt = f"""
Draft a press release from 10X Batteries Inc about its new product outlined in the YAML below. 
Emphasise that this product comes from ongoing development.
The reviews are from an earlier model.
Accenuate the postive sentiment from the reviews. 
Explain how critical feedback has been addressed.
{MOCK_PRODUCT}
"""

response = openai.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt, 
    max_tokens=150
)

print(response.choices[0].text.strip())

# %% [markdown]
# # 60 Automating Content Generation for Social Media Marketing

# %%
# 60.1 Mock up data
import yaml

# Mocked YAML data representing a social media stream
MOCK_STREAM = """
  - time: "2024-11-19T09:15:00Z"
    platform: "LinkedIn"
    channel: "Company Post"
    GUID: "post-1234"
    parent: null
    text: "Exciting updates from your company. Looking forward to seeing more!"

  - time: "2024-11-19T09:30:00Z"
    platform: "Facebook"
    channel: "Customer Feedback"
    GUID: "post-2234"
    parent: null
    text: "Is your product available in Canada?"

  - time: "2024-11-19T09:45:00Z"
    platform: "Twitter"
    channel: "Mentions"
    GUID: "post-3234"
    parent: null
    text: "Your battery is defective. I've been trying to contact support, but no response. Please help!"

  - time: "2024-11-19T10:00:00Z"
    platform: "Instagram"
    channel: "Comments on Reel"
    GUID: "post-4234"
    parent: null
    text: "Looks amazing! Can it be used in extreme weather conditions?"

  - time: "2024-11-19T10:15:00Z"
    platform: "LinkedIn"
    channel: "Product Announcement"
    GUID: "post-5234"
    parent: null
    text: "This is a game changer for off-grid energy solutions. Great work!"

  - time: "2024-11-19T10:30:00Z"
    platform: "Facebook"
    channel: "Customer Feedback"
    GUID: "post-6234"
    parent: "post-2234"
    text: "Thank you for your interest! We are planning to launch in Canada early next year."
"""

# Load the YAML data into Python dictionary
MOCK_STREAM = {m['GUID']:m for m in yaml.safe_load(MOCK_STREAM)}

# Display the data to verify
print(type(MOCK_STREAM), len(MOCK_STREAM))


# %%
# 60.2 Create the assistant
from openai import OpenAI 
from ipython_secrets import get_secret

# Set your API key
client = OpenAI(
    api_key=get_secret('OPENAI_API_KEY')
)

my_assistant = client.beta.assistants.create(
    instructions="""
    You will classify social media posts given to you in the form GUID: <text>. 
    You will tag each post as innocous (IGNORE), worthy of an AI response (AUTOREPLY) and flagged for human in the loop (HITL).
    Your response will be in the form GUID: tag
    """,
    name="Social Media Agent",
    model="gpt-4-turbo",
)
print(my_assistant)


# %%
# 60.3 Classifying Inputs

# Create a thread for the assistant
thread = client.beta.threads.create()

# Loop through the social media stream and send messages to classify each post
for post in MOCK_STREAM:
    post_text = post["text"]
    client.beta.threads.messages.create(
        thread_id=thread.id,
        content=f"{post['GUID']}: {post['text']}",
        role="user",
    )

run = client.beta.threads.runs.create_and_poll(
  thread_id=thread.id,
  assistant_id=my_assistant.id
)

if run.status == 'completed': 
  messages = client.beta.threads.messages.list(
    thread_id=thread.id
  )
  for m in messages:
    res = m.content[0].text.value
    print(res)

    for l in res.splitlines():
        k,v = l.split(':')
        MOCK_STREAM[k]['classification'] = v.strip()
    break

else:
  print(run.status)

# %%
# 60.4 Creating Responses
prompt = f"""
On behalf of 10X Batteries Inc, write a response to the following social media post: 
{MOCK_STREAM['post-2234']}
"""

response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt, 
    max_tokens=150
)

post_reply = response.choices[0].text.strip()
print(post_reply)

hitl_prompt = f"""
Write an SMS for Joe Bloggs, the Social Media manager of 10X Batteries Inc.
Joe must respond urgently to the following social media post copied below.
Include the GUID in the alert which will allow Joe to look up the post.
Refrain from pleasantries.
{MOCK_STREAM['post-3234']}
"""

response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=hitl_prompt, 
    max_tokens=300
)

hitl_msg = response.choices[0].text.strip()

print(hitl_msg)

# %%
# 60.5 Delete the thread and assistant

response = client.beta.threads.delete(thread.id)
print(response)

response = client.beta.assistants.delete(my_assistant.id)
print(response)

# %% [markdown]
# # 61 Sales Forecasting and Competitive Intelligence

# %%
# 61.1 Mocked Dataset: Historical Sales Data for 10X Batteries

import pandas as pd

# Create a DataFrame with mock sales data
SALES_DATA = pd.DataFrame({
    "Month": pd.date_range(start="2023-01-01", periods=12, freq="ME"),
    "Product": [
        "EcoBattery Pro 3000", "EcoBattery Pro 3000", "EcoBattery Pro 3000",
        "EcoBattery Lite 1500", "EcoBattery Lite 1500", "EcoBattery Lite 1500",
        "EcoBattery Max 5000", "EcoBattery Max 5000", "EcoBattery Max 5000",
        "EcoBattery Pro 3000", "EcoBattery Lite 1500", "EcoBattery Max 5000"
    ],
    "Units Sold": [120, 135, 150, 95, 110, 100, 60, 65, 80, 140, 105, 70],
    "Revenue (€)": [
        35970, 40425, 44950, 19950, 23100, 21000, 54000, 58500, 72000,
        41930, 22050, 63000
    ],
    "Region": [
        "North America", "North America", "Europe", "Europe", "Asia",
        "Asia", "North America", "North America", "Europe", "Europe",
        "Asia", "North America"
    ]
})

# Display the dataset
print(SALES_DATA)

# %%
# 61.2 Generate Forecasting Prompts from Mocked Sales Data

# Example prompt creation for GPT
def generate_forecasting_prompts(sales_data):
    prompts = []
    for product in sales_data["Product"].unique():
        product_data = sales_data[sales_data["Product"] == product]
        total_units = product_data["Units Sold"].sum()
        total_revenue = product_data["Revenue (€)"].sum()
        regions = product_data["Region"].unique()

        # Create a prompt for GPT analysis
        prompt = (
            f"Analyze the sales trends for the product '{product}'. "
            f"In the past year, it sold {total_units} units, generating €{total_revenue} in revenue. "
            f"The product was sold across these regions: {', '.join(regions)}. "
            "Provide insights into what might drive future sales, considering past performance and regional trends."
        )
        prompts.append(prompt)
    
    return prompts

# Generate prompts from the mock data
forecasting_prompts = generate_forecasting_prompts(SALES_DATA)

# Display generated prompts
for idx, prompt in enumerate(forecasting_prompts, 1):
    print(f"Prompt {idx}:\n{prompt}\n")

# %%
# 61.3: Analyze Sales Forecasts and Competitive Insights

import openai
from ipython_secrets import get_secret

# Set your API key
client = OpenAI(api_key=get_secret('OPENAI_API_KEY'))

# Function to generate forecasts and insights from GPT
def generate_forecasts_and_insights(query):
    prompt = f"Analyze the following query and provide insights based on past market trends and competition: {query}"

    # Send the prompt to the GPT model
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=500  # Customize based on desired response length
    )
    return response.choices[0].text.strip()

# Generate insights for each prompt
for idx, query in enumerate(forecasting_prompts, 1):
    print(f"Forecast and Insights for Prompt {idx}:\n")
    insights = generate_forecasts_and_insights(query)
    print(insights, "\n")

# %% [markdown]
# # 62 AI support of bid decision

# %%
# 62.1 Mock up a tender document
ITT_TEXT = """
Tender ID: TB-2024-001

Title: Supply and Installation of Energy Storage Systems

Green Energy Projects Ltd invites bids for the supply and installation of advanced energy storage systems to support its renewable energy initiative. This project will involve delivering high-quality energy storage solutions and ensuring their seamless integration with existing solar panel infrastructure.

Submission Deadline: All bids must be submitted no later than December 15, 2024.

Budget: The total budget for this project is €500,000.

Scope of Work:
	1.	The supplier is required to deliver and install 100 energy storage systems.
	2.	Each system must have a storage capacity of at least 3.0 kWh.
	3.	The energy storage systems must integrate with the existing solar panel infrastructure at the project site.
	4.	The supplier must provide training to local technicians for operation and maintenance of the systems.

Evaluation Criteria: Bids will be evaluated based on the following weighted criteria:
	•	Technical Capability: 30%
	•	Cost Competitiveness: 40%
	•	Delivery Timeline: 20%
	•	Sustainability Practices: 10%

Key Questions for Bidders:
	1.	Can the supplier guarantee delivery of all systems within six months of the contract award?
	2.	What sustainability certifications does the supplier hold?
	3.	Can the supplier provide a warranty of at least five years for the energy storage systems?
"""

# %%
# 62.2 Extract Key Information
from openai import OpenAI 
from ipython_secrets import get_secret

client = OpenAI(api_key=get_secret('OPENAI_API_KEY'))

# Standard form dictionary structure
DATA_DICT = {
    "Tender ID": None,
    "Title": None,
    "Issued By": None,
    "Submission Deadline": None,
    "Budget (€)": None,
    "Scope of Work": [],
    "Evaluation Criteria": {},
    "Questions": []
}

# Use OpenAI to extract and transform tender details into the dictionary format
def extract_tender_details(tender_text):
    prompt = f"""
    Analyze the following tender document and extract the key information into a structured dictionary format:
    {tender_text}
    Provide the result in this exact dictionary format:
    {{
        "Tender ID": "string",
        "Title": "string",
        "Issued By": "string",
        "Submission Deadline": "string",
        "Budget (€)": float,
        "Scope of Work": ["list of strings"],
        "Evaluation Criteria": {{"criteria name": "weightage in percentage as a string"}},
        "Questions": ["list of strings"]
    }}
    """
    
    # GPT API call
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=500
    )
    
    # Parse the response into Python dictionary
    extracted_details = eval(response.choices[0].text.strip())
    return extracted_details

# Extract details and load into standard form dictionary
ITT_DICT = extract_tender_details(ITT_TEXT)

# Display the resulting dictionary
print("Extracted Tender Details:")
for key, value in ITT_DICT.items():
    print(f"{key}: {value}\n" if isinstance(value, str) else f"{key}:\n{value}\n")


# %%
# 62.3 Balanced scorecard
# Define a function to evaluate a tender opportunity using a balanced scorecard
def evaluate_tender(tender, budget_weight=0.4, fit_weight=0.3, capability_weight=0.2, potential_weight=0.1):
    """
    Evaluate the tender opportunity based on a simple balanced scorecard approach.
    Weights are assigned to dimensions based on assumed priorities.

    Arguments:
    - tender: Dictionary containing tender details.
    - budget_weight, fit_weight, capability_weight, potential_weight: Relative weights for scoring.

    Returns:
    - A dictionary containing individual scores and the total score.
    """
    # Define scores for each dimension (mocked for simplicity; in practice, use calculations or AI insights)
    financial_viability = 8 if tender["Budget (€)"] > 200000 else 5  # Higher score for higher budgets
    customer_fit = 9 if len(tender["Scope of Work"]) >= 3 else 6  # Higher score for broader scope
    internal_capability = 7 if tender["Evaluation Criteria"].get("Technical Capability", 0) >= "30%" else 5
    future_potential = 6 if "sustainability" in str(tender).lower() else 4  # Check for strategic alignment

    # Weighted score
    total_score = (
        financial_viability * budget_weight +
        customer_fit * fit_weight +
        internal_capability * capability_weight +
        future_potential * potential_weight
    )

    # Return the balanced scorecard
    return {
        "Financial Viability": financial_viability,
        "Customer Fit": customer_fit,
        "Internal Capability": internal_capability,
        "Future Potential": future_potential,
        "Total Score": round(total_score, 2)
    }

# Evaluate the mocked tender document
balanced_scorecard = evaluate_tender(ITT_DICT)

# Display the scorecard
print("Balanced Scorecard Evaluation:")
for key, value in balanced_scorecard.items():
    print(f"{key}: {value}")


# %%
# 62.4 Assess Strategic Fit
from openai import OpenAI 
from ipython_secrets import get_secret

# Set your API key
client = OpenAI(api_key=get_secret('OPENAI_API_KEY'))

# Mocked historical tenders with balanced scorecards
historical_tenders = [
    {"Tender ID": "HT-2023-001", "Total Score": 6.5, "Strategic Outcome": "Neutral"},
    {"Tender ID": "HT-2023-002", "Total Score": 7.0, "Strategic Outcome": "Positive"},
    {"Tender ID": "HT-2023-003", "Total Score": 5.1, "Strategic Outcome": "Negative"},
    {"Tender ID": "HT-2023-004", "Total Score": 7.1, "Strategic Outcome": "Positive"},
]

# Current tender balanced scorecard
current_tender_scorecard = {
    "Tender ID": "TB-2024-001",
    "Total Score": 7.9,
    "Strategic Outcome": None,  # To be assessed
}

# Generate a narrative assessment of the opportunity using OpenAI
def generate_strategic_fit_assessment(current_tender, historical_tenders):
    prompt = f"""
    You are an analyst for 10X Batteries Inc. Evaluate the strategic fit of the following tender opportunity:
    Tender ID: {current_tender["Tender ID"]}
    Total Score: {current_tender["Total Score"]}

    Compare it with the historical tenders provided below:
    {historical_tenders}

    Assess the tender’s alignment with 10X Batteries' strategic objectives, including:
    - Expanding market share in renewable energy.
    - Aligning with sustainability goals.
    - Leveraging technical capabilities in energy storage systems.

    Provide a concise narrative explaining why this tender is a strong strategic fit and clearly recommend proceeding with a "yes."
    """

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

# Generate the strategic fit assessment
strategic_fit_assessment = generate_strategic_fit_assessment(current_tender_scorecard, historical_tenders)

# Display the assessment
print("Strategic Fit Assessment:\n")
print(strategic_fit_assessment)

# Send a request to the OpenAI API (> v1.0.0)
response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=f"Label the sentiment of: '{strategic_fit_assessment}' as POSTIVE, NEGATIVE or NEUTRAL",
    max_tokens=100
)

# Update the tender's strategic outcome
current_tender_scorecard["Strategic Outcome"] = response.choices[0].text.strip().upper()

# %%
# 62.5 Estimate Resource Requirements

import numpy as np

# Define parameters based on the tender document
tender_scope = {
    "units_required": 100,  # From Scope of Work
    "unit_capacity_kwh": 3.0,  # Minimum capacity per system
    "unit_cost": 4000,  # Estimated manufacturing cost per system (€)
    "installation_cost_per_unit": 500,  # Installation costs per unit (€)
    "training_cost": 2000,  # Fixed cost for technician training (€)
}

# Define historical operational capacities
historical_data = {
    "average_units_produced_per_month": 50,
    "available_months": 6,  # Delivery required in 6 months
    "current_team_size": 8,  # Number of operational staff
    "additional_team_cost_per_person": 3000,  # Monthly cost of hiring additional staff (€)
}

# Calculate estimated costs
def estimate_resource_requirements(tender_scope, historical_data):
    # Calculate production feasibility
    total_units = tender_scope["units_required"]
    max_production_capacity = historical_data["average_units_produced_per_month"] * historical_data["available_months"]
    requires_extra_capacity = total_units > max_production_capacity

    # Estimate costs
    manufacturing_cost = total_units * tender_scope["unit_cost"]
    installation_cost = total_units * tender_scope["installation_cost_per_unit"]
    training_cost = tender_scope["training_cost"]
    
    # Additional team costs if required
    additional_staff_needed = 0
    if requires_extra_capacity:
        additional_units = total_units - max_production_capacity
        additional_staff_needed = int(np.ceil(additional_units / (historical_data["available_months"] * historical_data["average_units_produced_per_month_per_person"])))
        additional_team_cost = additional_staff_needed * historical_data["additional_team_cost_per_person"] * historical_data["available_months"]
    else:
        additional_team_cost = 0

    total_cost = manufacturing_cost + installation_cost + training_cost + additional_team_cost

    # Return resource requirements
    return {
        "Manufacturing Cost (€)": manufacturing_cost,
        "Installation Cost (€)": installation_cost,
        "Training Cost (€)": training_cost,
        "Additional Team Cost (€)": additional_team_cost,
        "Total Cost (€)": total_cost,
        "Requires Additional Staff": requires_extra_capacity,
        "Additional Staff Needed": additional_staff_needed,
    }

# Estimate the resource requirements for the tender
resource_estimates = estimate_resource_requirements(tender_scope, historical_data)

# Display resource estimates
print("Resource Requirements Estimate:")
for key, value in resource_estimates.items():
    print(f"{key}: {value}")


# %%
# 52.6 Generate a Recommendation
# Strategic Assessment and Cost Inputs
strategic_fit = current_tender_scorecard["Strategic Outcome"]
total_cost = resource_estimates["Total Cost (€)"]
tender_budget = ITT_DICT["Budget (€)"]
expected_margin = tender_budget - total_cost

# Function to generate a pre-sales plan using AI
def generate_presales_plan(tender_text):
    prompt = f"""
    For the following tender, develop a pre-sales plan detailing the key activities and estimated costs. 
    Include tasks like bid preparation, client meetings, and technical clarifications. 
    The tender details are:
    {tender_text}
    Provide a breakdown of the pre-sales tasks and the total estimated cost in a structured format.
    """
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=300
    )
    return response.choices[0].text.strip()

# Generate the pre-sales plan
presales_plan = generate_presales_plan(ITT_TEXT)
print("Generated Pre-Sales Plan:\n", presales_plan)

# Mocked cost extraction for demonstration purposes
presales_cost = 15000  # Assume AI provided this cost as part of its output

# Generate Recommendation
def generate_recommendation(strategic_fit, expected_margin, presales_cost):
    # Define thresholds
    min_margin_threshold = 0.1 * tender_budget  # Minimum acceptable margin (10%)
    max_presales_threshold = 0.05 * tender_budget  # Maximum acceptable pre-sales cost (5%)
    
    # Logic for recommendation
    if strategic_fit == "Positive" and expected_margin >= min_margin_threshold and presales_cost <= max_presales_threshold:
        recommendation = "Bid"
        reasoning = (
            f"The opportunity aligns well with strategic goals, with a positive strategic assessment. "
            f"The expected margin of €{expected_margin:,} exceeds the minimum threshold of €{min_margin_threshold:,}, "
            f"and the pre-sales cost of €{presales_cost:,} is within the acceptable limit."
        )
    else:
        recommendation = "No Bid"
        reasoning = (
            f"The opportunity does not meet the criteria for a bid decision. Reasons include: "
            f"{'low margin' if expected_margin < min_margin_threshold else ''} "
            f"{'high pre-sales cost' if presales_cost > max_presales_threshold else ''}."
        )
    
    return recommendation, reasoning

# Generate final recommendation
recommendation, reasoning = generate_recommendation(strategic_fit, expected_margin, presales_cost)

# Display the recommendation
print("\nFinal Recommendation:")
print(f"Recommendation: {recommendation}")
print(f"Reasoning: {reasoning}")
