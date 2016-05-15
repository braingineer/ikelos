'''
author: bcm
'''


from __future__ import absolute_import, print_function
from keras.layers import Flatten
import keras.backend as K
from keras import activations, initializations, regularizers
from keras.engine import Layer, InputSpec


class Fix(Flatten):
    '''
    Flattens (up to but not including batch)
    Also affects the mask
    '''
    def __init__(self, return_mask=True, **kwargs):
        super(Fix, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_mask = return_mask

    def compute_mask(self, x, mask=None):
        if mask is None or not self.return_mask:
            return None
        return K.batch_flatten(mask)


class LambdaMask(Layer):
    '''
    muck up the mask, deliberately. 
    '''
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.supports_masking = True
        super(LambdaMask, self).__init__(*args, **kwargs)

    def compute_mask(self, x, mask=None):
        return self.func(x, mask)

    def call(self, x, mask=None):
        return x


class PassThrough(Layer):
    def __init__(self, *args, **kwargs):
        super(PassThrough, self).__init__(*args, **kwargs)
    def compute_mask(self, x, mask=None):
        return mask
    def call(self, x, mask=None):
        return x


class Compose(object):
    def __init__(self, layers, name=None):
        if name:
            layers = [PassThrough(name=name)]+layers
        self.layers = layers

    def get(self, i):
        if isinstance(self.layers[0], PassThrough):
            i+=1 
        return self.layers[i]

    def set_name(self, name):
        self.layers = [PassThrough(name=name)] + self.layers

    def __call__(self, x):
        out = x
        for layer in self.layers[::-1]:
            out = layer(out)
        return out

def compose(*layers):
    return Compose(layers)

def named_compose(name, *layers):
    return Compose(layers, name)

def set_name(tensor, name):
    return PassThrough(name=name)(tensor)
