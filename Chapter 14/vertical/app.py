from flask import Flask
import random
import string
import multiprocessing
import spacy
from timeit import default_timer as timer

app = Flask(__name__)

def generate_text(length):
    """Generate random text of specified length."""
    return "".join(random.choice(string.ascii_lowercase + " ") for _ in range(length))


def process_text(text):
    """Load an NLP model and process the text."""
    nlp = spacy.load("en_core_web_md")  # This is a medium-sized model
    doc = nlp(text)
    return len(doc.ents)  # Number of named entities in the text

def nlp_benchmark(n):
    """Parallel processing of NLP tasks."""
    texts = [generate_text(500) for _ in range(n)]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_text, texts)
    return sum(results)

@app.route("/")
def python_speed():
    total = 0
    result = {}

    # NLP benchmark
    start_time = timer()
    nlp_benchmark(5)  # Process n chunks of text
    elapsed_time = timer() - start_time
    total += elapsed_time
    result["NLP"] = round(elapsed_time * 1e3, 2)

    return result


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
