import torch
import torch.nn as nn
import numpy as np
import Transformer as t
from convokit import Corpus
from nltk.tokenize import word_tokenize
import json
import ujson
import os

import multihead_attention


def main():
    # home = os.path.expanduser('~')
    # path = ".convokit/downloads/"
    # corpus_name = "movie-corpus"
    # corpus = os.path.join(home, path, corpus_name)
    # file = os.path.join(corpus, "utterances.jsonl")
    #
    # with open(file, 'r') as datafile:
    #     lines = []
    #     for line in datafile:
    #         sentence_info = ujson.loads(line.rstrip())
    #
    #         reply_to = sentence_info['reply-to']
    #         text = sentence_info['text']
    #         new_parsed = []
    #         if (text != ""):
    #             new_parsed = word_tokenize(text)
    #         del sentence_info['meta']
    #         del sentence_info['timestamp']
    #         del sentence_info['vectors']
    #         del sentence_info['speaker']
    #         sentence_info.update({'parsed': new_parsed})
    #         lines.append(sentence_info)
    #     with open("preprocessed.json", "w") as outfile:
    #         json.dump(lines, outfile)
    #
    #
    #
    mask = torch.ones(4, 4)
    mask = torch.triu(mask, diagonal=1)
    x = torch.randn(2, 4, 2)
    print(x)
    print(mask)

    mh = multihead_attention.MultiHeadAttention(x, 2, mask)
    out = mh(x)
    print(out)

main()



