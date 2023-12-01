# Validate `instant-clip-tokenizer-py` by comparing its output with the original tokenizer on a
# large dataset.
#
# This script requires one command line argument, the filename of the data file to be used. The data
# file must be `.tsv` format with the strings to be used for validation in the first column.
#
# It is recommended to use this with Google's Conceptual Captions dataset (containing ~3.3M image
# captions), which can be downloaded here:
# https://ai.google.com/research/ConceptualCaptions/download

import html
import sys

import csv
import ftfy
import numpy as np

import instant_clip_tokenizer
import original


def validate(filename):
    tokenizer = instant_clip_tokenizer.Tokenizer()
    with open(filename) as f:
        rd = csv.reader(f, delimiter="\t", quotechar='"')
        for row in rd:
            text = row[0]
            expected = original.tokenize(text)
            ours = tokenizer.tokenize_batch(preprocess(text))
            if not np.array_equal(ours, expected):
                print(f"Failure: \"{text}\"")


def preprocess(text):
    # Copied from `basic_clean` function from `original.py`
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("validate.py must be called with exactly one argument - filename of data file", file=sys.stderr)
        sys.exit(-1)

    validate(sys.argv[1])
