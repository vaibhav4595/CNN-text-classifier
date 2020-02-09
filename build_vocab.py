from typing import List
from collections import Counter
from itertools import chain
from operator import itemgetter
import sys
import json
import pickle
import argparse
from utils import *
from pdb import set_trace as bp

class VocabEntry(object):
    def __init__(self, vocab_type='word'):
        self.word2id = dict()
        self.vocab_type = vocab_type
        self.unk_id = 2
        self.word2id['<pad>'] = 0
        self.word2id['<sep>'] = 1
        self.word2id['<unk>'] = 2
        self.total_class = 0
        self.class2id_dict = {}
        self.id2class_dict = {}
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def classIndex(self, class_set):
        for each in class_set:
            index = len(self.class2id_dict)
            self.class2id_dict[each] = index
            self.id2class_dict[index] = each

    def class2id(self, idx):
        return self.class2id_dict.get(idx, -1)

    def id2class(self, idx):
        return self.id2class_dict[idx]

    def class2indices(self, cids):
        return [self.class2id(idx) for idx in cids]

    def indices2class(self, cids):
        return [self.id2class(idx) for idx in cids]

    def numberize(self, sents):
        if self.vocab_type == 'word':
          return self.words2indices(sents)

    def denumberize(self, ids):
      if type(ids[0]) == list:
        if self.vocab_type == 'word':
          return [' '.join([self.id2word[w] for w in sent]) for sent in ids]
        else:
          return [''.join([self.id2word[w] for w in sent]) for sent in ids]
      else:
        if self.vocab_type == 'word':
          return ' '.join([self.id2word[w] for w in ids])
        else:
          return ''.join([self.id2word[w] for w in ids])

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def from_corpus(self, corpus, size, freq_cutoff=2):

        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print(f'number of word types: {len(word_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(valid_words)}')

        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            self.add(word)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', default='../data/topicclass/topicclass_train.txt')
    parser.add_argument('--vocab_size', type=int, default=70000)
    parser.add_argument('--save_name', default='vocab.pkl')

    args = parser.parse_args()

    corpus = []
    classes = set()
    print("Reading Input File")
    lines = open(args.input_file).readlines()

    for line in lines:
        tag, content = get_content(line)
        classes.add(tag)
        corpus.append(content)

    lens = [len(each) for each in corpus]
    print("Max Len = ", max(lens), "Min Len = ", min(lens), "Avg Len = ", sum(lens) / len(lens))
    vocab = VocabEntry()
    vocab.from_corpus(corpus, args.vocab_size)
    vocab.classIndex(classes)

    print("Dumping the vocabulary")
    fp = open(args.save_name, 'wb')
    pickle.dump(vocab, fp)
    fp.close()
