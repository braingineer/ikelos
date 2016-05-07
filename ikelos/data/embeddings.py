'''Load IGOR, make embeddings

Usage:
    embeddings.py convert <config_file>
    embeddings.py hash <config_file>

    convert <config_file>       convert data or a vocabulary to an embeddings file
                                based on one of the glove embeddigns
    hash <config_file>          experimental.  

author: bcm

goal: take as input a vocabulary and the location of an embedding file, then 
      instantiate a weight matrix which uses those embeddings + any random
      initializations for words not present in the existing embeddings.
k
preferred embeddings: glove. including a downloader. 

required igor settings:
- igor as data generator 
- embedding_size
    - should match target embeddings size
- target_glove
    - either name of url in glove urls or filepath
- from_url 
    - True if the target is a glove url
'''
from __future__ import absolute_import, print_function
from vocabulary import Vocabulary
from zipfile import ZipFile
import numpy as np
#from six.moves.urllib.parse import urlopen
import StringIO
from os import makedirs, path
from keras.initializations import glorot_uniform
from data_server import DataServer
from docopt import docopt
from tqdm import tqdm
import json
import os

class glove_urls:
    '''glove embeddings;

    Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.
    http://nlp.stanford.edu/projects/glove/
    '''

    ### Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download)
    glove_6b='http://nlp.stanford.edu/data/glove.6B.zip'
    ### Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download):
    glove_42b='http://nlp.stanford.edu/data/glove.42B.300d.zip'
    ### Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download):
    glove_840b='http://nlp.stanford.edu/data/glove.840B.300d.zip'
    ### Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download)
    glove_twitter_27b='http://nlp.stanford.edu/data/glove.twitter.27B.zip'



def from_generator(igor):
    print("getting vocab from data generator")
    vocab = Vocabulary.from_nlp_data(igor.generate(nb_times=1, source='train', flat=True))
    from_vocab(igor, vocab)


def from_vocab(igor, vocab):    
    print("using vocab and glove file to generate embedding matrix")
    remaining_vocab = set(vocab.keys())
    embeddings = np.zeros((len(vocab), igor.embedding_size))
    print("{} words to convert".format(len(remaining_vocab)))


    if igor.save_dir[-1] != "/":
        igor.save_dir += "/"
    if not path.exists(igor.save_dir):
        makedirs(igor.save_dir)

    if igor.from_url:
        assert hasattr(glove_urls, igor.target_glove), "You need to specify one of the glove variables"
        url = urlopen(getattr(glove_urls, igor.target_glove))
        fileiter = ZipFile(StringIO(url.read())).open(file).readlines()
    else:
        assert os.path.exists(igor.target_glove), "You need to specify a real file"
        fileiter = open(igor.target_glove).readlines()

    count=0
    for line in tqdm(fileiter):
        line = line.replace("\n","").split(" ")
        try:
            word, nums = line[0], [float(x.strip()) for x in line[1:]]
            if word in remaining_vocab:
                embeddings[vocab[word]]  = np.array(nums)
                remaining_vocab.remove(word)
        except Exception as e:
            print("{} broke. exception: {}. line: {}.".format(word, e, x))
        count+=1
    

    print("{} words were not in glove; saving to oov.txt".format(len(remaining_vocab)))
    with open(path.join(igor.save_dir, "oov.txt"), "w") as fp:
        fp.write("\n".join(remaining_vocab))

    for word in tqdm(remaining_vocab):
        embeddings[vocab[word]] = np.asarray(glorot_uniform((igor.embedding_size,)).eval())


    
    vocab.save('embedding.vocab')
    with open(path.join(igor.save_dir, "embedding.npy"), "wb") as fp:
        np.save(fp, embeddings)

def make_hash_embeddings(igor, vocab):
    assert os.path.exists(igor.target_glove), "You need to specify a real file"
    fileiter = open(igor.target_glove).readlines()

    hash_vocab = Vocabulary()
    hash_vocab.use_mask = True
    hash_vocab.add(hash_vocab.mask_symbol)
    hash_vocab.add(hash_vocab.unk_symbol)
    word2hash = {}
    for word, v_id in vocab.items():
        ids = hash_vocab.add_many(hash_word(word))
        word2hash[v_id] = ids

    embeddings = np.zeros((len(hash_vocab), igor.embedding_size))
    remaining_vocab = set(vocab.keys())
    remaining_hashes = set(hash_vocab.values())
    for line in tqdm(fileiter):
        line = line.replace("\n","").split(" ")
        word, nums = line[0], [float(x.strip()) for x in line[1:]]
        word_hash = hash_word(word)
        if word in remaining_vocab:
            hash_ids = word2hash[vocab[word]]
            remaining_vocab.remove(word)
            remaining_hashes.difference_update(hash_ids)
            embeddings[hash_ids] += np.array(nums) / len(hash_ids)
    print("{} words were not seen.  {} hashes were not seen".format(len(remaining_vocab),
                                                                    len(remaining_hashes)))
    for hash_id in remaining_hashes:
        embeddings[hash_id] = np.asarray(glorot_uniform((igor.embedding_size,)).eval())

    glove_name = igor.target_glove[igor.target_glove.find("glove"):].replace("/","")

    hash_vocab.save('hash_embedding_{}.vocab'.format(glove_name))
    with open(path.join(igor.save_dir, "hash_embedding_{}.npy".format(glove_name)), "wb") as fp:
        np.save(fp, embeddings)
    with open(path.join(igor.save_dir, "word2hash.json".format(glove_name)), "w") as fp:
        json.dump(word2hash, fp)


def hash_word(word):
    wleft = lambda w: "##"+word
    wmid = lambda w: "#"+word+"#"
    wright = lambda w: word+"##"
    return ["".join(letters) for letters in zip(wleft(word), wmid(word), wright(word))]

if __name__ == "__main__":
    '''
    pipeline: 
        1. take as input the sequence generator
        2. go through once, get all of the words
        3. make the vocabulary
        4. get the selected embedding
        5. iterate over them, taking the words and discarding the ones i don't need
        6. save the embeddings and the vocabulary
    seperately, also need:
        1. a convert dataset with known vocabulary
        2. something that gets the embeddings and greats the embedding matrix with them
    '''
    args = docopt(__doc__, version='conversion script; May 2016')
    igor = DataServer.from_file(args['<config_file>'])
    if args['hash']:
        make_hash_embeddings(igor, Vocabulary.load(igor.vocab_file))
    elif args['convert']:
        if igor.vocab_file:
            from_vocab(igor, Vocabulary.load(igor.vocab_file))
        else:
            from_generator(igor)




'''TODO: convert this ruby script for preprocessing tweets:

# Ruby 2.0
# Reads stdin: ruby -n preprocess-twitter.rb
#
# Script for preprocessing tweets by Romain Paulus
# with small modifications by Jeffrey Pennington

def tokenize input

    # Different regex parts for smiley faces
    eyes = "[8:=;]"
    nose = "['`\-]?"

    input = input
        .gsub(/https?:\/\/\S+\b|www\.(\w+\.)+\S*/,"<URL>")
        .gsub("/"," / ") # Force splitting words appended with slashes (once we tokenized the URLs, of course)
        .gsub(/@\w+/, "<USER>")
        .gsub(/#{eyes}#{nose}[)d]+|[)d]+#{nose}#{eyes}/i, "<SMILE>")
        .gsub(/#{eyes}#{nose}p+/i, "<LOLFACE>")
        .gsub(/#{eyes}#{nose}\(+|\)+#{nose}#{eyes}/, "<SADFACE>")
        .gsub(/#{eyes}#{nose}[\/|l*]/, "<NEUTRALFACE>")
        .gsub(/<3/,"<HEART>")
        .gsub(/[-+]?[.\d]*[\d]+[:,.\d]*/, "<NUMBER>")
        .gsub(/#\S+/){ |hashtag| # Split hashtags on uppercase letters
            # TODO: also split hashtags with lowercase letters (requires more work to detect splits...)

            hashtag_body = hashtag[1..-1]
            if hashtag_body.upcase == hashtag_body
                result = "<HASHTAG> #{hashtag_body} <ALLCAPS>"
            else
                result = (["<HASHTAG>"] + hashtag_body.split(/(?=[A-Z])/)).join(" ")
            end
            result
        } 
        .gsub(/([!?.]){2,}/){ # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
            "#{$~[1]} <REPEAT>"
        }
        .gsub(/\b(\S*?)(.)\2{2,}\b/){ # Mark elongated words (eg. "wayyyy" => "way <ELONG>")
            # TODO: determine if the end letter should be repeated once or twice (use lexicon/dict)
            $~[1] + $~[2] + " <ELONG>"
        }
        .gsub(/([^a-z0-9()<>'`\-]){2,}/){ |word|
            "#{word.downcase} <ALLCAPS>"
        }

    return input
end

puts tokenize($_)
'''