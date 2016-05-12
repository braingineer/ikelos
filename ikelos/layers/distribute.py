from __future__ import absolute_import, print_function
from keras.layers import Recurrent, time_distributed_dense, LSTM, Wrapper, TimeDistributed
import keras.backend as K
from keras.engine import Layer, InputSpec


Distribute = TimeDistributed

class Summarize(Wrapper):
    def __init__(self, summarizer, *args, **kwargs):
        self.supports_masking = True
        self.last_two = None
        super(Summarize, self).__init__(summarizer, *args, **kwargs)

    def build(self, input_shape=None):
        '''Assumes that self.layer is already set.
        Should be called at the end of .build() in the
        children classes.
        '''        
        ndim = len(input_shape)
        assert ndim >= 3
        self.input_spec = [InputSpec(ndim=str(ndim)+'+')]
        #if input_shape is not None:
        #    self.last_two = input_shape[-2:]
        self._input_shape = input_shape
        #self.input_spec = [InputSpec(shape=input_shape)]
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



        #child_input_shape = (np.prod(input_shape[:-2]),) + input_shape[-2:]
        child_input_shape = (None,)+input_shape[-2:]
        if not self.layer.built:
            self.layer.build(child_input_shape)
            self.layer.built = True

        self.trainable_weights = getattr(self.layer, 'trainable_weights', [])
        self.non_trainable_weights = getattr(self.layer, 'non_trainable_weights', [])
        self.updates = getattr(self.layer, 'updates', [])
        self.regularizers = getattr(self.layer, 'regularizers', [])
        self.constraints = getattr(self.layer, 'constraints', {})

    def get_output_shape_for(self, input_shape):
        child_input_shape = (1,) + input_shape[-2:]
        child_output_shape = self.layer.get_output_shape_for(child_input_shape)
        return input_shape[:-2] + child_output_shape[-1:]

    def compute_mask(self, x, mask=None):
        if mask is None:
            return None
        #import pdb
        #pdb.set_trace()
        target_dim = K.ndim(x) - 2
        num_reducing = K.ndim(mask) - target_dim
        if num_reducing:
            axes = tuple([-i for i in range(1,num_reducing+1)])
            mask = K.any(mask, axes)

        return mask

    def call(self, x, mask=None):
        if hasattr(x, '_keras_shape'):
            input_shape = x._keras_shape
        else:
            input_shape = self._input_shape
        #import pdb
        #pdb.set_trace()
        #if self.last_two is not None:
        #    last2 = self.last_two
        #else:
        #    input_shape = x._keras_shape
        #    last2 = input_shape[-2:]
        #out_shape = K.shape(x)[:-2]

        x = K.reshape(x, (-1,) + input_shape[-2:]) # (batch * d1 * ... * dn-2, dn-1, dn)
        if mask is not None:
            mask_shape = (K.shape(x)[0], -1)
            mask = K.reshape(mask, mask_shape) # give it the same first dim
        y = self.layer.call(x, mask)
        #try:
        #output_shape = self.get_output_shape_for(K.shape(x))
        #except:
        output_shape =  self.get_output_shape_for(input_shape)
        #import pdb
        #pdb.set_trace()
        return K.cast(K.reshape(y, output_shape), K.floatx()) 

    def get_config(self):
        config = {}  #'summary_space_size': self.summary_space_size
        base_config = super(Summarize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class LastDimDistribute(Wrapper):
    def __init__(self, distributee, *args, **kwargs):
        self.supports_masking = True
        super(LastDimDistribute, self).__init__(distributee, *args, **kwargs)

    def build(self, input_shape=None):
        '''Assumes that self.layer is already set.
        Should be called at the end of .build() in the
        children classes.
        '''        
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]
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



        child_input_shape = (np.prod(input_shape[:-1]),) + input_shape[-1:]
        if not self.layer.built:
            self.layer.build(child_input_shape)
            self.layer.built = True

        self.trainable_weights = getattr(self.layer, 'trainable_weights', [])
        self.non_trainable_weights = getattr(self.layer, 'non_trainable_weights', [])
        self.updates = getattr(self.layer, 'updates', [])
        self.regularizers = getattr(self.layer, 'regularizers', [])
        self.constraints = getattr(self.layer, 'constraints', {})

    def get_output_shape_for(self, input_shape):
        child_input_shape = (1,) + input_shape[-1:]
        child_output_shape = self.layer.get_output_shape_for(child_input_shape)
        return input_shape[:-1] + child_output_shape[-1:]

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        x = K.reshape(x, (-1,) + input_shape[-1:]) # (batch * d1 * ... * dn-2*dn-1, dn)
        mask_shape = (K.shape(x)[0], -1)
        mask = K.reshape(mask, mask_shape) # give it the same first dim
        y = self.layer.call(x, mask)
        output_shape = self.get_output_shape_for(input_shape)
        return K.reshape(y, output_shape)

    def get_config(self):
        config = {}  #'summary_space_size': self.summary_space_size
        base_config = super(LastDimDistribute, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

