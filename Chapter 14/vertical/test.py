import requests
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vertical scaling client script")

    parser.add_argument("iter", help="Number of iterations")

    args = parser.parse_args()

    the_range = range(int(args.iter))

    for i in tqdm(the_range):
        r = requests.get("http://localhost:4000")

    for i in tqdm(the_range):
        r = requests.get("http://localhost:4001")

    for i in tqdm(the_range):
        r = requests.get("http://localhost:4002")

    for i in tqdm(the_range):
        r = requests.get("http://localhost:4003")
