



def make_embeddings(igor, vocab):
    #glove_fp = "/research/data/glove/glove.840B.300d.txt"
    vocab_togo = set(vocab._mapping.keys())
    E = igor.embedding_size
    V = len(vocab)
    embeddings = np.zeros((V, E), dtype=np.float32)
    with open(igor.glove_fp) as fp:
        for line in tqdm(fp.readlines(), unit='lines'):
            line = line.replace("\n","").split(" ")
            word,nums = line[0], [float(x.strip()) for x in line[1:]]
            if word in vocab_togo:
                vocab_togo.remove(word)
                embeddings[vocab[word]] = np.array(nums)

    print("{} words not in glove".format(len(vocab_togo)))
    with open("glove_oov.txt", "w") as fp:
        fp.write("\n".join(vocab_togo))

    for word in vocab_togo:
        embeddings[vocab[word]] = glorot_uniform((E,)).eval()

    utils.save_embeddings(igor, embeddings, vocab)
    return embeddings
