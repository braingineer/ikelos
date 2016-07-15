'''
testing to see if the trees are actually working
'''


'''
test set 1:
    root: any number
    children: any other number
    grandchild: multiplication of first two numbers along path
'''


'''
test set 2:
    root: any number
    children: any other number, with earlier children 
    ???
'''

import keras
import ikelos
import numpy as np
import theano 
import theano.tensor as T
from copy import copy
import random
import itertools
from keras.layers import Input
from keras.engine import Model
from ikelos.layers import RTTN



def mix(a,b):
    a,b = copy(a), copy(b)
    o = []
    while len(a)>0 and len(b)>0:
        if random.random()>0.5:
            o.append(a.pop())
        else:
            o.append(b.pop())
    if len(a)>0:
        o.extend(a)
    else:
        o.extend(b)
    return o

def generate_data_single(seq_size, domain=30):
    while True:
        offset = random.randint(0,10)
        xbase = np.linspace(0+offset,domain+offset,seq_size+1)
        xsin = np.sin(xbase)
        xcos = np.cos(xbase)

        ysin, xsin = xsin[1:], xsin[:-1]
        ycos, xcos = xcos[1:], xcos[:-1]

        dsin = zip(xsin,ysin,[0]*len(xsin))
        dcos = zip(xcos,ycos,[1]*len(xcos))

        mixed = mix(dsin, dcos)
        xdata = np.array([x[0] for x in mixed])
        ydata = np.array([x[1] for x in mixed])
        tdata = np.array([x[2] for x in mixed]).astype(np.int32)
        yield [xdata, tdata], ydata

def generate_data_batch(batch_size, seq_size, domain=30):
    if seq_size % 2:
        print("error. for now, seq size has to be even. adjusting")
        seq_size += 1
    dataiter = generate_data_single(seq_size // 2, domain)
    while True:
        xdata = np.zeros((batch_size, seq_size,1), dtype=theano.config.floatX)
        tdata = np.zeros((batch_size, seq_size), dtype=np.int32)
        ydata = np.zeros((batch_size, seq_size,1), dtype=theano.config.floatX)
        for i, ((x,t), y) in enumerate(itertools.islice(dataiter, 0, batch_size)):
            xdata[i] = x
            ydata[i] = y
            tdata[i] = t
        yield [xdata, tdata], ydata


def test1():
    seq_size = 10
    batch_size = 10 
    rnn_size = 1
    xin = Input(batch_shape=(batch_size, seq_size,1))
    xtop = Input(batch_shape=(batch_size, seq_size))
    xbranch, xsummary = RTTN(rnn_size, return_sequences=True)([xin, xtop])

    model = Model(input=[xin, xtop], output=[xbranch, xsummary])
    model.compile(loss='MSE', optimizer='SGD')
    data_gen = generate_data_batch(batch_size, seq_size)
    model.fit_generator(generator=data_gen, samples_per_epoch=1000, nb_epoch=100)

test1()



