
from __future__ import print_function, division
import yaml
import time
import sys
import numpy as np
import itertools
from ..utils import loggers
from keras.utils.np_utils import to_categorical
try:
    input = raw_input
except:
    pass


class DataServer(object):
    """
    Required config:
    saving_prefix
    batch_size

    Is also good to "stage" it by adding to it the following:
        train_data
        dev_data
        train_vocab
    """
    def __init__(self, config):
        self.__dict__.update(config)


    def print_everything(self):
        for key, value in self.__dict__.items():
            if isinstance(value, (bool, str, int, float)):
                print("{}: {}".format(key, value))

    @classmethod
    def from_file(cls, config_file):
        with open(config_file) as fp:
            config = yaml.load(fp)
        return cls(config)

    @property
    def num_train_batches(self):
        return len(self.train_data)//self.batch_size

    @property
    def num_dev_batches(self):
        return len(self.dev_data)//self.batch_size
        
    @property
    def num_train_samples(self):
        num_samples = self.num_train_batches * self.batch_size
        if self.subepochs > 0:
            reduced_number = num_samples // self.subepochs
            self.logger.info("+ {} subepochs reduces ".format(self.subepochs) + 
                             "  {} per epoch to {}".format(num_samples, 
                                                           reduced_number))
            return reduced_number
        return num_samples
        
    @property
    def num_dev_samples(self):
        return self.num_dev_batches * self.batch_size

    def stage(self, *args, **kwargs):
        """ would be great to load things here """
        pass

    def serve_single(self, data):
        """serve a single sample from a dataset; 
           yield X and Y of size appropriate to sample_size

        ### an example implementation
        for data_i in np.random.choice(len(data), len(data), replace=False):
            in_X = np.zeros(self.max_sequence_len)
            out_Y = np.zeros(self.max_sequence_len, dtype=np.int32)
            bigram_data = zip(data[data_i][0:-1], data[data_i][1:])
            for datum_j,(datum_in, datum_out) in enumerate(bigram_data):
                in_X[datum_j] = datum_in
                out_Y[datum_j] = datum_out
            yield in_X, out_Y
        """
        raise NotImplementedError 

    def serve_batch(self, data):
        """serve a batch of samples from a dataset; 
           yield X and Y of sizes appropriate to (batch,) + sample_size 

        ### an example implementation
        ### yields (batch,sequence) and (batch, sequence, vocab)
        dataiter = self.serve_sentence(data)
        V = self.vocab_size
        S = self.max_sequence_len
        B = self.batch_size

        while dataiter:
            in_X = np.zeros((B, S), dtype=np.int32)
            out_Y = np.zeros((B, S, V), dtype=np.int32)
            next_batch = list(itertools.islice(dataiter, 0, self.batch_size))
            if len(next_batch) < self.batch_size:
                raise StopIteration
            for d_i, (d_X, d_Y) in enumerate(next_batch):
                in_X[d_i] = d_X
                out_Y[d_i] = to_categorical(d_Y, V)
                
            yield in_X, out_Y
        """
        raise NotImplementedError

    def _data_gen(self, data, forever=True):
        working = True
        while working:
            for batch in self.serve_batch(data):
                yield batch
            working = working and forever
        
    def dev_gen(self, forever=True):
        return self._data_gen(self.dev_data, forever)

    def train_gen(self, forever=True):
        return self._data_gen(self.train_data, forever)            
    