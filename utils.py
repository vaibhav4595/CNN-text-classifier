import os
import sys
import math
import random
import pickle
from build_vocab import VocabEntry
from pdb import set_trace as bp

def get_content(line):

    line = line.strip()
    tag, content = line.split("|||")
    tag = tag.strip()
    content = content.strip().lower()
    content = content.split()
    for i, each in enumerate(content):
        if each.startswith('@') and each.endswith('@'):
            content[i] = each[1:-1]

    return tag, content

def get_vocab(filename):

    print("Loading Vocabulary")
    fp = open(filename, 'rb')
    vocab = pickle.load(fp)

    return vocab

def batch_iter(lines, vocab, batch_size, max_sent_len, shuffle=False):

    batch_num = math.ceil(len(lines) / batch_size)

    index_array = list(range(len(lines)))

    content = []
    for each in lines:
        tag, data = get_content(each)
        data = vocab.words2indices(data)

        # pad here
        data = data[0:max_sent_len] + [0] * (max_sent_len - len(data))

        tag = vocab.class2id(tag)
        content.append([tag, data])

    if shuffle == True:
        random.shuffle(content)

    for i in range(batch_num):

        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [content[idx][1] for idx in indices]
        labels = [content[idx][0] for idx in indices]

        yield examples, labels

