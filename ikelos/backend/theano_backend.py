from keras.backend import *
import theano
import theano.tensor as T
import numpy as np


def stack_rnn(step_function, inputs, initial_states, stack_indices, 
                             go_backwards=False, 
                             mask=None, constants=None, unroll=False,
                             input_length=None):

    ndim = inputs.ndim
    assert ndim >= 3, 'Input should be at least 3D.'

    axes = [1, 0] + list(range(2, ndim))
    inputs = inputs.dimshuffle(axes)
    stack_indices = stack_indices.dimshuffle([1,0])

    if constants is None:
        constants = []

    if mask is not None:
        if mask.ndim == ndim-1:
            mask = expand_dims(mask)
        assert mask.ndim == ndim
        mask = mask.dimshuffle(axes)


        if unroll:
            indices = list(range(input_length))
            if go_backwards:
                indices = indices[::-1]

            successive_outputs = []
            state_tensors = initial_states
            initial = [state0[0,:,:] for state0 in state_tensors]
            batch_index = T.arange(state_tensors[0].shape[1])
            prev_output = parent_state = None
            for x_ind in indices:
                p_ind = stack_indices[x_ind]
                if parent_state is None:
                    parent_state = initial
                else:
                    parent_state = [state_tensor[p_ind, batch_index] for state_tensor in state_tensors]
                output, new_states = step_function(inputs[x_ind], parent_state + constants)

                if prev_output is None:
                    prev_output = zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = T.switch(mask[x_ind], output, prev_output)
                kept_states = []
                for i,(p_state, new_state) in enumerate(zip(parent_state, new_states)):
                    state_tensors[i] = T.set_subtensor(state_tensors[i][x_ind], 
                                                       T.switch(mask[x_ind], p_state, new_state))
                #state_stack.append(kept_states)
                successive_outputs.append(output)

            outputs = T.stack(*successive_outputs) 
            htensor, ctensor = state_tensors
            #states = []
            #for i in range(len(state_stack[-1])):
            #    states.append(T.stack(*[states_at_step[i] for states_at_step in state_stack]))
        else:
        
            # build an all-zero tensor of shape (samples, output_dim)
            init_states = [s[0,:,:] for s in initial_states]
            initial_output = step_function(inputs[0], init_states + constants)[0] * 0
            # Theano gets confused by broadcasting patterns in the scan op
            initial_output = T.unbroadcast(initial_output, 0, 1)
            def _step(input, mask, stack_index, iter_index, output_tm1, h_tensor, c_tensor, *constants):
                batch_index = T.arange(stack_index.shape[0])
                hm1 = colgather(h_tensor, batch_index, stack_index) 
                #hm1 = h_tensor[stack_index]
                cm1 = colgather(c_tensor, batch_index, stack_index)
                #cm1 = c_tensor[stack_index]
                output, [h, c] = step_function(input, [hm1, cm1]+list(constants))
                output = T.switch(mask, output, output_tm1)

                assert mask.ndim == h.ndim == c.ndim == hm1.ndim == cm1.ndim
                h = T.switch(mask, h, hm1)
                c = T.switch(mask, c, cm1)
                return [output, T.set_subtensor(h_tensor[iter_index], h), 
                                T.set_subtensor(c_tensor[iter_index], c)]

            (outputs, htensor, ctensor), _ = theano.scan(
                                            _step,
                                            sequences=[inputs, 
                                                       mask, 
                                                       stack_indices, 
                                                       T.arange(inputs.shape[0])],
                                            outputs_info=[initial_output]+initial_states,
                                            non_sequences=constants,
                                            go_backwards=go_backwards)     

            htensor = htensor[-1]
            ctensor = ctensor[-1]   

        
    else:

        if unroll:
            indices = list(range(input_length))
            if go_backwards:
                indices = indices[::-1]

            successive_outputs = []
            state_tensors = initial_states
            initial = [state0[0,:,:] for state0 in state_tensors]
            prev_output = parent_state = None
            for x_ind in indices:
                p_ind = stack_indices[x_ind]
                if parent_state is None:
                    parent_state = initial
                else:
                    parent_state = [state_tensor[p_ind] for state_tensor in state_tensors]
                output, new_states = step_function(inputs[x_ind], parent_state + constants)

                if prev_output is None:
                    prev_output = zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                for i, state in enumerate(new_states):
                    state_tensors[i] = T.set_subtensor(state_tensors[i][x_ind], state)

                successive_outputs.append(output)

            outputs = T.stack(*successive_outputs)
            htensor, ctensor = state_tensors
            #states = []
            #for i in range(len(state_stack[-1])):
            #    states.append(T.stack(*[states_at_step[i] for states_at_step in state_stack]))

        else:


            def _step(input, stack_index, iter_index, h_tensor, c_tensor, *constants):
                batch_index = T.arange(stack_index.shape[0])
                hm1 = colgather(h_tensor, batch_index, stack_index) 
                #hm1 = h_tensor[stack_index]
                cm1 = colgather(c_tensor, batch_index, stack_index)
                #cm1 = c_tensor[stack_index]
                output, [h, c] = step_function(input, [hm1, cm1]+list(constants))
                return [output, T.set_subtensor(h_tensor[iter_index], h), 
                                T.set_subtensor(c_tensor[iter_index], c)]

            (outputs, htensor, ctensor), _ = theano.scan(
                                            _step,
                                            sequences=[inputs,
                                                       stack_indices, 
                                                       T.arange(inputs.shape[0])],
                                            outputs_info=[None]+initial_states,
                                            non_sequences=constants,
                                            go_backwards=go_backwards)     

            htensor = htensor[-1]
            ctensor = ctensor[-1]   



    outputs = T.squeeze(outputs)
    last_output = outputs[-1]

    axes = [1, 0] + list(range(2, outputs.ndim))
    outputs = outputs.dimshuffle(axes)
    return last_output, outputs, [T.squeeze(htensor), T.squeeze(ctensor)]



def dualsignal_rnn(step_function, inputs, initial_states, stack_indices, 
                                  go_backwards=False, 
                                  mask=None, constants=None, unroll=False,
                                  input_length=None):
 
    ndim = inputs.ndim
    assert ndim >= 3, 'Input should be at least 3D.'

    axes = [1, 0] + list(range(2, ndim))
    inputs = inputs.dimshuffle(axes)
    stack_indices = stack_indices.dimshuffle([1,0])

    if constants is None:
        constants = []

    if mask is not None:
        if mask.ndim == ndim-1:
            mask = expand_dims(mask)
        assert mask.ndim == ndim
        mask = mask.dimshuffle(axes)


        if unroll:
            indices = list(range(input_length))
            if go_backwards:
                indices = indices[::-1]

            successive_outputs = []
            state_tensors = initial_states
            initial = [state0[0,:,:] for state0 in state_tensors]
            batch_index = T.arange(state_tensors[0].shape[1])
            prev_output = parent_state = None
            for x_ind in indices:
                p_ind = stack_indices[x_ind]
                if parent_state is None:
                    parent_state = initial
                else:
                    parent_state = [state_tensor[p_ind, batch_index] for state_tensor in state_tensors]
                output, new_states = step_function(inputs[x_ind], parent_state + constants)

                if prev_output is None:
                    prev_output = zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = T.switch(mask[x_ind], output, prev_output)
                kept_states = []
                for i,(p_state, new_state) in enumerate(zip(parent_state, new_states)):
                    state_tensors[i] = T.set_subtensor(state_tensors[i][x_ind], 
                                                       T.switch(mask[x_ind], p_state, new_state))
                #state_stack.append(kept_states)
                successive_outputs.append(output)

            outputs = T.stack(*successive_outputs)
            htensor, ctensor = state_tensors
            #states = []
            #for i in range(len(state_stack[-1])):
            #    states.append(T.stack(*[states_at_step[i] for states_at_step in state_stack]))
        else:
        
            # build an all-zero tensor of shape (samples, output_dim)
            init_states = [(s[0,:,:], s[0,:,:]) for s in initial_states]
            initial_output = step_function(inputs[0], init_states + constants)[0] * 0
            # Theano gets confused by broadcasting patterns in the scan op
            initial_output = T.unbroadcast(initial_output, 0, 1)
            def _step(input, mask, stack_index, iter_index, head_tm1, summary_tm1, 
                                   h_tensor, c_tensor, hs_tm1, cs_tm1, *constants):
                batch_index = T.arange(stack_index.shape[0])
                hm1 = colgather(h_tensor, batch_index, stack_index) 
                #hm1 = h_tensor[stack_index]
                cm1 = colgather(c_tensor, batch_index, stack_index)
                #cm1 = c_tensor[stack_index]
                states = [(hm1, hs_tm1), (cm1, cs_tm1)]+list(constants)
                head_out, [h, c] = step_function(input, states)
                head_out = T.switch(mask, head_out, head_tm1)

                assert mask.ndim == h.ndim == c.ndim == hm1.ndim == cm1.ndim
                h = T.switch(mask, h, hm1)
                c = T.switch(mask, c, cm1)

                s_states = [(hs_tm1, h), (cs_tm1, c)] + list(constants)
                sum_out, [hs,cs] = step_function(input, s_states)
                sum_out = T.switch(mask, sum_out, summary_tm1)
                hs = T.switch(mask, hs, hs_tm1)
                cs = T.switch(mask, cs, cs_tm1)
                return [head_out, sum_out, T.set_subtensor(h_tensor[iter_index], h), 
                                           T.set_subtensor(c_tensor[iter_index], c), 
                                           hs, cs]

            output_info = [initial_output, initial_output] # so we can retrieve the head_tm1 and summary_tm1
            output_info += initial_states # so we can track the tree tensor
            output_info += [s[0] for x in init_states]  # so we can get the  summary info

            (tree_outputs, summary_outputs,
             htensor, ctensor, hs, cs), _ = theano.scan(
                                             _step,
                                            sequences=[inputs, 
                                                       mask, 
                                                       stack_indices, 
                                                       T.arange(inputs.shape[0])],
                                            outputs_info=output_info,
                                            non_sequences=constants,
                                            go_backwards=go_backwards)     

            ctensor = ctensor[-1]   
            htensor = htensor[-1]
        
    else:

        if unroll:
            indices = list(range(input_length))
            if go_backwards:
                indices = indices[::-1]

            successive_outputs = []
            state_tensors = initial_states
            initial = [state0[0,:,:] for state0 in state_tensors]
            prev_output = parent_state = None
            for x_ind in indices:
                p_ind = stack_indices[x_ind]
                if parent_state is None:
                    parent_state = initial
                else:
                    parent_state = [state_tensor[p_ind] for state_tensor in state_tensors]
                output, new_states = step_function(inputs[x_ind], parent_state + constants)

                if prev_output is None:
                    prev_output = zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                for i, state in enumerate(new_states):
                    state_tensors[i] = T.set_subtensor(state_tensors[i][x_ind], state)

                successive_outputs.append(output)

            outputs = T.stack(*successive_outputs)
            htensor, ctensor = state_tensors
            #states = []
            #for i in range(len(state_stack[-1])):
            #    states.append(T.stack(*[states_at_step[i] for states_at_step in state_stack]))

        else:
            def _step(input, stack_index, iter_index, h_tensor, c_tensor, hs_tm1, cs_tm1, *constants):
                batch_index = T.arange(stack_index.shape[0])
                hm1 = colgather(h_tensor, batch_index, stack_index) 
                cm1 = colgather(c_tensor, batch_index, stack_index)

                # in the order of (me, other)
                states = [(hm1, hs_tm1), (cm1, cs_tm1)]+list(constants)
                head_out, [h, c] = step_function(input, states)

                s_states = [(hs_tm1, h), (cs_tm1, c)] + list(constants)
                sum_out, [hs,cs] = step_function(input, s_states)
                
                return [head_out, sum_out, T.set_subtensor(h_tensor[iter_index], h), 
                                           T.set_subtensor(c_tensor[iter_index], c),
                                           hs, cs]

            output_info = [None, None]+initial_states + [x[0,:,:] for x in initial_states]
            (tree_outputs, summary_outputs, 
             htensor, ctensor, hc, cs), _ = theano.scan(
                                                _step,
                                                sequences=[inputs,
                                                           stack_indices, 
                                                           T.arange(inputs.shape[0])],
                                                outputs_info=output_info,
                                                non_sequences=constants,
                                                go_backwards=go_backwards)     

            htensor = htensor[-1]
            ctensor = ctensor[-1]   


    tree_outputs = T.squeeze(tree_outputs)
    summary_outputs = T.squeeze(summary_outputs)
    last_tree = tree_outputs[-1]
    last_summary = summary_outputs[-1]
    
    axes = [1, 0] + list(range(2, tree_outputs.ndim))
    tree_outputs = tree_outputs.dimshuffle(axes)
    summary_outputs = summary_outputs.dimshuffle(axes)
    return (last_tree, last_summary), (tree_outputs, summary_outputs), [T.squeeze(htensor), T.squeeze(ctensor)]




def rttn(step_function, inputs, initial_states, tree_topology, action_types, 
                                horizon, shape_key, context_matrix, 
                                mask=None, constants=None, **kwargs):
 
    assert inputs.ndim >= 3, 'Input should be at least 3D.'

    horizon_words, horizon_indices = horizon

    _shuffle = lambda tensor: tensor.dimshuffle([1,0]+list(range(2,tensor.ndim))) 
    inputs = _shuffle(inputs)
    tree_topology = _shuffle(tree_topology)
    action_types = _shuffle(action_types)
    horizon_words = _shuffle(horizon_words) # all words on horizon
    horizon_indices = _shuffle(horizon_indices) # all of their branch indices


    if constants is None:
        constants = []

    if mask is not None:
        if mask.ndim == inputs.ndim-1:
            mask = expand_dims(mask)
        assert mask.ndim == inputs.ndim
        mask = _shuffle(mask)

        def _step(iter_index, x_input, x_mask, x_type, x_topology, 
                              horizon_words, horizon_indices, 
                              h_traverse, branch_tensor, W_ctx, *constants):
            '''Notes for this function:
               W_ctx is passed in under non sequences but is separated here from the constants
            '''
            ### topology
            batch_index = T.arange(x_topology.shape[0])
            h_parent = colgather(branch_tensor, batch_index, x_topology)
            states = (h_parent, h_traverse, x_type) + constants
            h_child, h_vplus = step_function(x_input, states)
            ## is masking necessary for branches? idk.
            h_child = T.switch(x_mask, h_child, h_parent)
            h_vplus = T.switch(x_mask, h_vplus, h_traverse)

            branch_tensor = T.set_subtensor(branch_tensor[iter_index], h_child)


            ### shape sizes
            s_batch = shape_key['batch']
            s_rnn = shape_key['rnn'] 
            s_word = shape_key['word']
            s_rnn_word = s_rnn + s_word

            # ctx is used as an attentional vector over the horizon states
            # W_ctx is (R, RW), h_vplus is (B, R); horizont_types is (B,)
            # horizon_types lets different tree actions be considered
            ctx = T.dot(h_vplus, W_ctx)
            
            ctx_shape = (s_batch, 1, s_rnn_word)
            ctx = T.reshape(ctx, ctx_shape)
            T.addbroadcast(ctx, 1)

            # horizon state is (B, HorizonSize, RNNWORD)
            branch_owners = branch_tensor[horizon_indices, T.arange(s_batch).reshape((s_batch, 1))]
            #branch_owners = branch_tensor[T.arange(s_batch), horizon_indices] # indexes into the branches
            horizon_state = T.concatenate([branch_owners, horizon_words], axis=-1) 
            
            # now create the probability tensor
            p_horizon = horizon_state * ctx  # elemwise multiplying
            p_horizon = T.sum(p_horizon, axis=-1) #then summing. 
            #this was basically a dot, but per batch row and resulting in a dim reduction
            # now, given (B,Horizon), we can get a softmax distribution per row
            p_horizon = T.nnet.softmax(p_horizon)
            # note, this means we can also sample if we want to do a dynamic oracle. 
            
            return h_vplus, branch_tensor, horizon_state, p_horizon

        output_info = initial_states + [None, None]
        
        (h_v, branch_tensor, 
         horizon_states, p_horizons), _ = theano.scan(
                                                 _step,
                                                sequences=[T.arange(inputs.shape[0]), 
                                                           inputs, 
                                                           mask, 
                                                           action_types,
                                                           tree_topology, 
                                                           horizon_words, 
                                                           horizon_indices],
                                                outputs_info=output_info,
                                                non_sequences=[context_matrix] + constants)     
        branch_tensor = branch_tensor[-1]
        
    else:

        def _step(iter_index, x_input, x_type, x_topology, 
                              horizon_words, horizon_indices, 
                              h_traverse, branch_tensor, W_ctx, *constants):
            '''Notes for this function:
               W_ctx is passed in under non sequences but is separated here from the constants
            '''
            ### topology
            batch_index = T.arange(x_topology.shape[0])
            h_parent = colgather(branch_tensor, batch_index, x_topology)
            states = (h_parent, h_traverse, x_type) + constants
            h_child, h_vplus = step_function(x_input, states)
            
            branch_tensor = T.set_subtensor(branch_tensor[iter_index], h_child)
            
            ### shape sizes
            s_batch = shape_key['batch']
            s_rnn = shape_key['rnn'] 
            s_word = shape_key['word']
            s_rnn_word = s_rnn + s_word

            # ctx is used as an attentional vector over the horizon states
            # W_ctx is (4, R, RW), h_vplus is (B, R); horizont_types is (B,)
            # horizon_types lets different tree actions be considered
            ctx = T.dot(h_vplus, W_ctx)
            ctx = T.addbroadcast(T.reshape(ctx, (s_batch, 1, s_rnn_word)), 1)

            # horizon state is (B, HorizonSize, s_rnn_word)
            branch_owners = branch_tensor[T.arange(s_batch), horizon_indices] # indexes into the branches
            horizon_state = T.concatenate([branch_owners, horizon_words], axis=-1) 
            
            # now create the probability tensor
            p_horizon = horizon_state * ctx  # elemwise multiplying
            p_horizon = T.sum(p_horizon, axis=-1) #then summing. 
            #this was basically a dot, but per batch row and resulting in a dim reduction
            # now, given (B,Horizon), we can get a softmax distribution per row
            p_horizon = T.nnet.softmax(p_horizon) # b, horizon
            #p_horizon = T.addbroadcast(T.reshape(p_horizon, (s_batch, s_horizon, 1)), 1)
            # note, this means we can also sample if we want to do a dynamic oracle. 
            #horizon_attn = T.sum(p_horizon * horizon_state, axis=1)
            
            
            return h_vplus, branch_tensor, horizon_state, p_horizon

        output_info = initial_states + [None, None]
        
        (h_v, branch_tensor, 
         horizon_states, p_horizons), _ = theano.scan(
                                                 _step,
                                                sequences=[T.arange(inputs.shape[0]), 
                                                           inputs, 
                                                           action_types,
                                                           tree_topology, 
                                                           horizon_words, 
                                                           horizon_indices],
                                                outputs_info=output_info,
                                                non_sequences=[context_matrix] + constants)     
        branch_tensor = branch_tensor[-1]
    
    unshuffle = lambda tensor: T.squeeze(tensor).dimshuffle([1, 0] + list(range(2, tensor.ndim)))
    h_v = unshuffle(h_v)
    branch_tensor = unshuffle(branch_tensor)
    horizon_states = unshuffle(horizon_states)
    p_horizons = unshuffle(p_horizons)

    return branch_tensor, h_v, horizon_states, p_horizons