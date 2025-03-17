from flask import Flask, request
from transformers import pipeline
import random
from data import TEXT

app = Flask(__name__)
# Load Hugging Face pipeline
sentiment_analyzer = pipeline("sentiment-analysis", framework="pt")


@app.route("/", methods=["GET", "POST"])
def nlp_jobs():
    if request.method == "POST":
        data = request.json
        input = data["text"]
    else:
        input = random.choice(TEXT)

    sentiment = sentiment_analyzer(input)

    return {
        "input": input,
        "sentiment": sentiment,
    }


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
