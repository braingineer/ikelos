from keras.callbacks import Callback
from sklearn.metrics.pairwise import cosine_similarity

class SpineSampler(Callback):
    def __init__(self, spine_embedder, igor):
        self.spine_embedder = spine_embedder
        self.igor = igor

    def on_epoch_end(self, epoch, logs={}):
        indices = np.random.choice(len(self.spine_embedder), 10, False)
        comparisons = cosine_similarity(self.spine_embedder[indices], self.spine_embedder)
        results = np.argmax(comparisons, axis=-1)
        spine_vocab = self.igor.vocabs.spines
        comp_spines = [spine_vocab.lookup(x) for x in results]
        in_spines = [spine_vocab.lookup(x) for x in indices]
        for spine_i, spine_j in zip(in_spines, comp_spines):
            print("SPINE: {}".format(self.decode(spine_i)))
            print("\t most similar to {}".format(self.decode(spine_j)))
        
