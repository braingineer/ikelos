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



def compose(*layers):
    def func(x):
        out = x 
        for layer in layers[::-1]:
            out = layer(out)
        return out
    return func