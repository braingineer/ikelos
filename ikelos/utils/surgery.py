'''
perform regular surgery operations
'''

from keras.engine import Model
import pprint
pp = pprint.PrettyPrinter().pprint

def crop_to_layer(model, layer):
    inputs = model.inputs
    assert len(layer.inbound_nodes)==1
    outputs = layer.inbound_nodes[0].output_tensors
    #print(inputs)
    path = [layer]
    seen = set()
    necessary = []
    while len(path)>0:
        #print(len(path), len(seen))
        nxt = path.pop()
        #print(nxt.name, nxt.name in seen, nxt)
        for node in nxt.inbound_nodes:
            layers = [layer for layer in node.inbound_layers if layer.name not in seen]
            seen.update(set([layer.name for layer in layers]))
            path.extend(layers)
            for tensor in node.input_tensors:
                if tensor in inputs and tensor not in necessary:
                    necessary.append(tensor)

        #pp(nxt.__dict__)
    return necessary, outputs

def crop_to_tensor(model, tensor):
    layer = tensor._keras_history[0]
    return crop_to_layer(model, layer)


def crop(model, layer_or_tensor):
    if hasattr(layer_or_tensor, '_keras_history'):
        ins, outs = crop_to_tensor(model, layer_or_tensor)
    else:
        ins, outs = crop_to_layer(model, layer_or_tensor)
    return Model(ins, outs, preloaded_data=model.preloaded_data)