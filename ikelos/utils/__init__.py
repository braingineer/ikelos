import keras.backend as K
from keras.engine import merge
from ..layers import PassThrough
import loggers

def normalize_mask(x, mask):
    '''Keep the mask align wtih the tensor x

    Arguments: x is a data tensor; mask is a binary tensor
    Rationale: keep mask at same dimensionality as x, but only with a length-1 
               trailing dimension. This ensures broadcastability, which is important
               because inferring shapes is hard and shapes are easy to get wrong. 
    '''
    mask = K.cast(mask, K.floatx())
    while K.ndim(mask) != K.ndim(x):
        if K.ndim(mask) > K.ndim(x):
            mask = K.any(mask, axis=-1)
        elif K.ndim(mask) < K.ndim(x):
            mask = K.expand_dims(mask)
    return K.any(mask, axis=-1, keepdims=True)


concat = lambda x: merge(x, mode='concat')
def xor(a,b, v=None):
    return (a is not v and b is v) or (a is v and b is not v)
