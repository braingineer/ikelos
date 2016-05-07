'''
recurrent tree networks

author: bcm
'''


from __future__ import absolute_import, print_function
from keras.layers import Recurrent, time_distributed_dense, LSTM
import keras.backend as K
from keras import activations, initializations, regularizers
from keras.engine import Layer, InputSpec



class DualCurrent(Recurrent):
    ''' modified from keras's lstm; the recurrent tree network
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(DualCurrent, self).__init__(**kwargs)

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.permute_dimensions(x, [1,0,2]) # (timesteps, samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (timesteps, samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def build(self, input_shapes):
        assert isinstance(input_shapes, list)
        rnn_shape, indices_shape = input_shapes
        self.input_spec = [InputSpec(shape=rnn_shape), InputSpec(shape=indices_shape)]
        input_dim = rnn_shape[2]
        self.input_dim = input_dim

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]

        ''' add a second incoming recurrent connection '''

        self.W_i = self.init((input_dim, self.output_dim),
                             name='{}_W_i'.format(self.name))
        self.U_i_me = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_i_me'.format(self.name))
        self.U_i_other = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_i_other'.format(self.name))
        self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

        self.W_f = self.init((input_dim, self.output_dim),
                             name='{}_W_f'.format(self.name))
        self.U_f_me = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_f_me'.format(self.name))
        self.U_f_other = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_f_other'.format(self.name))
        self.b_f = self.forget_bias_init((self.output_dim,),
                                         name='{}_b_f'.format(self.name))

        self.W_c = self.init((input_dim, self.output_dim),
                             name='{}_W_c'.format(self.name))
        self.U_c_me = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_c_me'.format(self.name))
        self.U_c_other = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_c_other'.format(self.name))
        self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

        self.W_o = self.init((input_dim, self.output_dim),
                             name='{}_W_o'.format(self.name))
        self.U_o_me = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_o_me'.format(self.name))
        self.U_o_other = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_o_other'.format(self.name))
        self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(K.concatenate([self.W_i,
                                                        self.W_f,
                                                        self.W_c,
                                                        self.W_o]))
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(K.concatenate([self.U_i_me,self.U_i_other, 
                                                        self.U_f_me,self.U_f_other,
                                                        self.U_c_me,self.U_c_other,
                                                        self.U_o_me,self.U_o_other]))
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(K.concatenate([self.b_i,
                                                        self.b_f,
                                                        self.b_c,
                                                        self.b_o]))
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W_i, self.U_i_me, self.U_i_other, self.b_i,
                                  self.W_c, self.U_c_me, self.U_c_other, self.b_c,
                                  self.W_f, self.U_f_me, self.U_f_other, self.b_f,
                                  self.W_o, self.U_o_me, self.U_o_other, self.b_o]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[1], input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[1], input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[1], input_shape[0], self.output_dim)),
                           K.zeros((input_shape[1], input_shape[0], self.output_dim))]

    def compute_mask(self, input, mask):
        if self.return_sequences:
            if isinstance(mask, list):
                return [mask[0], mask[0]]
            return [mask, mask]
        else:
            return [None, None]
        
    def get_output_shape_for(self, input_shapes):
        rnn_shape, indices_shape = input_shapes
        out_shape = super(DualCurrent, self).get_output_shape_for(rnn_shape)
        return [out_shape, out_shape]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            if 0 < self.dropout_W < 1:
                dropout = self.dropout_W
            else:
                dropout = 0
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_i = time_distributed_dense(x, self.W_i, self.b_i, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_f = time_distributed_dense(x, self.W_f, self.b_f, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_c = time_distributed_dense(x, self.W_c, self.b_c, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_o = time_distributed_dense(x, self.W_o, self.b_o, dropout,
                                         input_dim, self.output_dim, timesteps)
            return K.concatenate([x_i, x_f, x_c, x_o], axis=2)
        else:
            return x

    def step(self, x, states):
        (h_tm1_me, h_tm1_other) = states[0]
        (c_tm1_me, c_tm1_other) = states[1]
        B_U = states[2]
        B_W = states[3]

        if self.consume_less == 'cpu':
            x_i = x[:, :self.output_dim]
            x_f = x[:, self.output_dim: 2 * self.output_dim]
            x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
            x_o = x[:, 3 * self.output_dim:]
        else:
            x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
            x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
            x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
            x_o = K.dot(x * B_W[3], self.W_o) + self.b_o

        i = self.inner_activation(x_i + K.dot(h_tm1_me * B_U[0], self.U_i_me) + 
                                        K.dot(h_tm1_other * B_U[0], self.U_i_other))

        f_me = self.inner_activation(x_f + K.dot(h_tm1_me * B_U[1], self.U_f_me) + 
                                           K.dot(h_tm1_other * B_U[1], self.U_f_me))
        f_other = self.inner_activation(x_f + K.dot(h_tm1_me * B_U[1], self.U_f_other) +
                                              K.dot(h_tm1_other * B_U[1], self.U_f_other))
        
        in_c = i * self.activation(x_c + K.dot(h_tm1_me * B_U[2], self.U_c_me) + 
                                         K.dot(h_tm1_other * B_U[2], self.U_c_other))
        re_c = f_me * c_tm1_me + f_other * c_tm1_other
        c = in_c + re_c

        o = self.inner_activation(x_o + K.dot(h_tm1_me * B_U[3], self.U_o_me) + 
                                        K.dot(h_tm1_other * B_U[3], self.U_o_other))

        h = o * self.activation(c)
        return h, [h, c]

    def call(self, xpind, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        x, indices = xpind
        if isinstance(mask, list):
            mask, _ = mask
        input_shape = self.input_spec[0].shape
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
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.dualsignal_rnn(self.step, 
                                                   preprocessed_input,
                                                   initial_states, 
                                                   indices,
                                                   go_backwards=self.go_backwards,
                                                   mask=mask,
                                                   constants=constants,
                                                   unroll=self.unroll,
                                                   input_length=input_shape[1])

        last_tree, last_summary = last_output
        tree_outputs, summary_outputs = outputs

        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))
            self.cached_states = states

        return [tree_outputs, summary_outputs]

        

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(DualCurrent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class StackLSTM(LSTM):
    def build(self, input_shapes):
        assert isinstance(input_shapes, list)
        rnn_shape, indices_shape = input_shapes
        super(StackLSTM, self).build(rnn_shape)
        self.input_spec += [InputSpec(shape=indices_shape)]

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.permute_dimensions(x, [1,0,2]) # (timesteps, samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (timesteps, samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[1], input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[1], input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[1], input_shape[0], self.output_dim)),
                           K.zeros((input_shape[1], input_shape[0], self.output_dim))]
    
    def get_output_shape_for(self, input_shapes):
        rnn_shape, indices_shape = input_shapes
        return super(StackLSTM, self).get_output_shape_for(rnn_shape)

    def compute_mask(self, input, mask):
        if self.return_sequences:
            if isinstance(mask, list):
                return mask[0]
            return mask
        else:
            return None
    
    def call(self, xpind, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        x, indices = xpind
        if isinstance(mask, list):
            mask, _ = mask
        input_shape = self.input_spec[0].shape
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
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.stack_rnn(self.step, 
                                                   preprocessed_input,
                                                   initial_states, 
                                                   indices,
                                                   go_backwards=self.go_backwards,
                                                   mask=mask,
                                                   constants=constants,
                                                   unroll=self.unroll,
                                                   input_length=input_shape[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))
            self.cached_states = states

        if self.return_sequences:
            return outputs
        else:
            return last_output
