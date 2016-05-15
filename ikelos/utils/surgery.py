'''
perform regular surgery operations
'''


def crop_to_layer(model, layer):
    inputs = model.inputs
    path = [layer]
    necessary = []
    while len(path)>0:
        nxt = path.pop()
        if nxt in inputs:
            necessary.append(nxt)
        for node in nxt.inbound_nodes:
            path.extend(node.inbound_layers)
    return necessary

def crop_to_tensor(model, tensor):
    layer = tensor._keras_history[0]
    return crop_to_layer(model, layer)