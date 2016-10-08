"""
a set of attention layers for keras and science
"""

from __future__ import absolute_import, print_function
from keras.layers import Dense, Wrapper, Distribute
import keras.backend as K
from keras import activations, initializations, regularizers
from keras.engine import Layer, InputSpec
#from keras.activations import softmax
import numpy as np

def make_safe(x):
    return K.clip(x, K.common._EPSILON, 1.0 - K.common._EPSILON)


def softmax(x):
    e = K.exp(x - K.max(x, axis=-1, keepdims=True))
    s = K.sum(e, axis=-1, keepdims=True)
    return e / s
    
class ProbabilityTensor(Wrapper):
    """ function for turning 3d tensor to 2d probability matrix """
    def __init__(self, dense_function=None, *args, **kwargs):
        self.supports_masking = True
        if dense_function is None:
            dense_function = Dense(1, name='ptensor_func')
        layer = Distribute(dense_function)
        super(ProbabilityTensor, self).__init__(layer, *args, **kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis.')

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        super(ProbabilityTensor, self).build()

    def get_output_shape_for(self, input_shape):
        # b,n,f -> b,n 
        #       s.t. \sum_n n = 1
        return (input_shape[0], input_shape[1])

    def squash_mask(self, mask):
        if K.ndim(mask) == 2:
            return mask
        elif K.ndim(mask) == 3:
            return K.all(mask, axis=-1)
        return mask

    def compute_mask(self, x, mask=None):
        return None

    def call(self, x, mask=None):
        energy = K.squeeze(self.layer(x), 2)
        p_matrix = softmax(energy)
        if mask is not None:
            mask = self.squash_mask(mask)
            p_matrix = make_safe(p_matrix * mask) # remove unwanted items
            p_matrix = p_matrix / K.sum(p_matrix, axis=-1, keepdims=True) # renormalize
        return make_safe(p_matrix)


class SoftAttention(ProbabilityTensor):
    '''
    Standard.  make probability distribution, weight, normalize. 
    '''
    def __init__(self, return_probabilities=False, *args, **kwargs):
        super(SoftAttention, self).__init__(*args, **kwargs)
        self.return_probabilities = return_probabilities

    def get_output_shape_for(self, input_shape):
        # b,n,f -> b,f where f is weighted features summed across n
        return (input_shape[0], input_shape[2])

    def compute_mask(self, x, mask=None):
        if mask is None or mask.ndim==2:
            return None
        elif mask.ndim==3:
            mask = K.any(mask, axis=(1,2))
        else:
            raise Exception("Unexpected situation")

    def call(self, x, mask=None):
        # b,n,f -> b,f via b,n broadcasted
        p_vectors = K.expand_dims(super(SoftAttention, self).call(x, mask), 2)
        expanded_p = K.repeat_elements(p_vectors, K.shape(x)[2], axis=2)
        attended = K.sum(expanded_p * x, axis=1)
        if self.return_probabilities:
            return [attended, p_vectors]
        return attended

    def get_config(self):
        config = {'return_probabilities': self.return_probabilities}
        base_config = super(SoftAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class EZAttend(Layer):
    '''
    if you needed the probability distribution for loss purposes, have no fear!
    you can just put it into the input here and use it for making an attended vector =)
    '''
    def __init__(self, p_tensor, *args, **kwargs):
        self.supports_masking = True
        self.p_tensor = p_tensor
        super(EZAttend, self).__init__(*args, **kwargs)

    def compute_mask(self, x, mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        last_dim = K.ndim(self.p_tensor)
        output_shape = list(input_shape)
        output_shape.pop(last_dim-1)
        return tuple(output_shape)

    def call(self, target_tensor, mask=None):
        last_dim = K.ndim(self.p_tensor)
        expanded_p = K.repeat_elements(K.expand_dims(self.p_tensor, last_dim), 
                                       K.shape(target_tensor)[last_dim], 
                                       axis=last_dim)
        return K.sum(expanded_p * target_tensor, axis=last_dim-1)


class Accumulator(SoftAttention):
    def get_output_shape_for(self, input_shape):
        ## this won't actually change the shape. it'll just accumulate up to t =)
        assert len(input_shape) == 3
        if self.return_probabilities:
            pvec_shape = (input_shape[0], input_shape[1], input_shape[1])
            return [input_shape, pvec_shape]    
        return input_shape

    def build(self, input_shape):
        super(Accumulator, self).build(input_shape)
        ## setting this explicilty so we can use it 
        self.input_spec = [InputSpec(shape=input_shape)]

    def compute_mask(self, x, mask=None):
        if self.return_probabilities:
            mask2 = mask
            if mask is not None:
                mask2 = K.expand_dims(K.all(mask2, axis=-1))
            return [mask, mask2]
        return mask

    def attend_function(self, inputs, mask=None):
        # b,n,f -> b,f via b,n broadcasted
        inputs = K.permute_dimensions(inputs, (1,0,2)) ### assuming it comes from an unroller
        if mask:
            mask = K.permute_dimensions(mask, (1,0,2))
        output = super(Accumulator, self).call(inputs, mask)
        return output

    def call(self, x, mask=None):
        ''' assuming a 3dim tensor, batch,time,feat '''
        input_length = self.input_spec[0].shape[1]
        results = accumulate(self.attend_function, x, input_length,
                             mask=mask, return_probabilities=self.return_probabilities)
        return results



def accumulate(attend_function, inputs, input_length,
                                mask=None, return_probabilities=False):
    '''get the running attention over a sequence. 

    given a 3dim tensor where the 1st dim is time (or not. whatever.),  calculating the running attended sum.
    in other words, at the first time step, you only have that item.
                    at the second time step, attend over the first two items.
                    at the third..  the third. so on. 

    this basically a mod on keras' rnn implementation
    author: bcm
    '''

    ndim = inputs.ndim
    assert ndim >= 3, 'inputs should be at least 3d'

    axes = [1,0] + list(range(2, ndim))
    inputs = inputs.dimshuffle(axes)

    indices = list(range(input_length))

    successive_outputs = []
    if mask is not None:
        if mask.ndim == ndim-1:
            mask = K.expand_dims(mask)
        assert mask.ndim == ndim
        mask = mask.dimshuffle(axes)
        prev_output = None

    successive_outputs = []
    successive_pvecs = []
    uncover_mask = K.zeros_like(inputs)
    uncover_indices = K.arange(input_length)
    for _ in range(ndim-1):
        uncover_indices = K.expand_dims(uncover_indices)
    make_subset = lambda i,X: K.switch(uncover_indices <= i, X, uncover_mask)
    for i in indices:
        inputs_i = make_subset(i,inputs)
        mask_i = make_subset(i,mask)
        if mask is not None:
            output = attend_function(inputs_i, mask_i) # this should not output the time dimension; it should be marginalized over. 
        else:
            output = attend_function(inputs_i) # this should not output the time dimension; it should be marginalized over. 
        if return_probabilities:
            output, p_vectors = output
            successive_pvecs.append(p_vectors)
        assert output.ndim == 2, "Your attention function is malfunctioning; the attention accumulator should return 2 dimensional tensors"
        successive_outputs.append(output)
    outputs = K.pack(successive_outputs)
    K.squeeze(outputs, -1)
    axes = [1, 0] + list(range(2, outputs.ndim))
    outputs = outputs.dimshuffle(axes)

    if return_probabilities:
        out_pvecs = K.pack(successive_pvecs)
        K.squeeze(out_pvecs, -1)
        out_pvecs = out_pvecs.dimshuffle(axes)
        outputs = [outputs, out_pvecs]

    return outputs