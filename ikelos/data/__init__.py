from __future__ import absolute_import
from .vocabulary import Vocabulary
from .vocab_manager import VocabManager
from .data_server import DataServer
from .. import utils

import os 
try:
    import cPickle as pickle
except:
    import pickle

from collections import Counter
import json


def load_sentence_dataset(filepath, add_period=True, max_length=float('inf')):
    f = lambda x: [y.lower() for y in x.replace("\n","").split(" ")] + (["."] if "." not in x[-4:] and add_period else [])
    stats = [0,0]
    out = []
    with open(filepath) as fp:
        if "pkl" in filepath[-4:]:
            dataset = pickle.load(fp)
        elif ".txt" in filepath[-4:]:
            dataset = fp.readlines()
        else:
            raise Exception("unknown dataset file")
        for sent in dataset:
            datum = f(sent)
            if len(datum) <= max_length:
                stats[0] += 1
                out.append(datum)
            else:
                stats[1] += 1
        print("data file: {}. {} kept; {} discarded.".format(filepath, stats[0], stats[1]))
        return out



def to_vocab(data, frequency_cutoff=None, size_cutoff=None):
    if not utils.xor(frequency_cutoff, size_cutoff):
        raise Exception("one or the other cutoffs please")

    counter = Counter(word for sent in data for word in sent)

    if frequency_cutoff is not None:    
        print("Using a frequency of {} to reduce vocabulary size.".format(frequency_cutoff))
        words = [word for word,count in counter.most_common() if count > frequency_cutoff]
        print("Vocabulary size reduced. {} -> {}".format(len(counter), len(words)))
        
    elif size_cutoff is not None:
        print("Using a cutoff of {} to reduce vocabulary size.".format(size_cutoff))
        words = [word for word,count in counter.most_common(size_cutoff)]
        print("Vocabulary size reduced. {} -> {}".format(len(counter), len(words)))
    
    else:
        raise Exception("should never happen...")
    
    vocab = Vocabulary(use_mask=True)
    vocab.add_many(['<START>', "<END>"])
    vocab.add_many(words)
    return vocab
