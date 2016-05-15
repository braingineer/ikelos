from keras import backend as K
from keras.layers import Embedding
import numpy as np



class DynamicEmbedding(Embedding):
    def __init__(self, embedding_matrix, mode='matrix', *args, **kwargs):
        assert hasattr(embedding_matrix, '_keras_shape')
        self.W = embedding_matrix
        if mode=='tensor':
            assert len(embedding_matrix._keras_shape) == 3
            indim = self.W._keras_shape[1]
            outdim = self.W._keras_shape[2]
        else:
            assert len(embedding_matrix._keras_shape) == 2
            indim, outdim = self.W._keras_shape

        self.mode = mode
        super(DynamicEmbedding, self).__init__(indim, outdim, *args, **kwargs)

        #layer, node_index, tensor_index = self.W._keras_history
        #self.add_inbound_node(layer, node_index, tensor_index)
        
        
    def __call__(self, x, mask=None):
        ### hacky. 
        return super(DynamicEmbedding, self).__call__([x, self.W], mask)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape,_ = input_shape

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            
        self.built = True

    def compute_mask(self, x, mask=None):
        if isinstance(x, list):
            x,_ = x
        if mask is not None and isinstance(mask, list):
            mask,_ = mask
        return super(DynamicEmbedding, self).compute_mask(x, mask)

    def get_output_shape_for(self, input_shape):
        if isinstance(input_shape, list):
            input_shape,_ = input_shape
        return super(DynamicEmbedding, self).get_output_shape_for(input_shape)

    def call(self, x, mask=None):
        if isinstance(x, list): 
            x,_ = x
        if mask is not None and isinstance(mask, list):
            mask,_ = mask
        if 0. < self.dropout < 1.:
            retain_p = 1. - self.dropout
            dims = self.W._keras_shape[:-1]
            B = K.random_binomial(dims, p=retain_p) * (1. / retain_p)
            B = K.expand_dims(B)
            W = K.in_train_phase(self.W * B, self.W)
        else:
            W = self.W
        
        if self.mode == 'matrix':
            return K.gather(W,x)
        elif self.mode == 'tensor':
            # quick and dirty: only allowing for 3dim inputs when it's tensor mode
            assert K.ndim(x) == 3
            # put sequence on first; gather; take diagonal across shared batch dimension
            # in other words, W is (B, S, F)
            # incoming x is (B, S, A)
            inds = K.arange(self.W._keras_shape[0])
            #out = K.gather(K.permute_dimensions(W, (1,0,2)), x).diagonal(axis1=0, axis2=3)
            #return K.permute_dimensions(out, (3,0,1,2))
            ### method above doesn't do grads =.=
            # tensor abc goes to bac, indexed onto with xyz, goes to xyzac, 
            # x == a, so shape to xayzc == xxyzc
            # take diagonal on first two: xyzc 
            #out = K.colgather()
            out = K.gather(K.permute_dimensions(W, (1,0,2)), x) 
            out = K.permute_dimensions(out, (0,3,1,2,4))
            out = K.gather(out, (inds, inds))
            return out
        else:
            raise Exception('sanity check. should not be here.')

        #all_dims = T.arange(len(self.W._keras_shape))
        #first_shuffle = [all_dims[self.embed_dim]] + all_dims[:self.embed_dim] + all_dims[self.embed_dim+1:]
        ## 1. take diagonal from 0th to
        ## chang eof tactics
        ## embed on time or embed on batch. that's all I'm supporting.  
        ## if it's embed on time, then, x.ndim+1 is where batch will be, and is what
        ## i need to take the diagonal over. 
        ## now dim shuffle the xdims + 1 to the front.
        #todo: get second shuffle or maybe find diagonal calculations
        #out = K.gather(W, x)
        #return out

        ### reference
        #A = S(np.arange(60).reshape(3,4,5))
        #x = S(np.random.randint(0, 4, (3,4,10)))
        #x_emb = A.dimshuffle(1,0,2)[x].dimshuffle(0,3,1,2,4)[T.arange(A.shape[0]), T.arange(A.shape[0])]