from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np

def train_word2vec(sentence_matrix, vocabulary_inv, num_features=300, min_word_count=1, context=10):
    if False:
        for i in range(10):
            print('nop')
    '\n    Trains, saves, loads Word2Vec model\n    Returns initial weights for embedding layer.\n   \n    inputs:\n    sentence_matrix # int matrix: num_sentences x max_sentence_len\n    vocabulary_inv  # dict {str:int}\n    num_features    # Word vector dimensionality                      \n    min_word_count  # Minimum word count                        \n    context         # Context window size \n    '
    model_dir = 'word2vec_models'
    model_name = '{:d}features_{:d}minwords_{:d}context'.format(num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print("Loading existing Word2Vec model '%s'" % split(model_name)[-1])
    else:
        num_workers = 2
        downsampling = 0.001
        print('Training Word2Vec model...')
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)
        embedding_model.init_sims(replace=True)
        if not exists(model_dir):
            os.mkdir(model_dir)
        print("Saving Word2Vec model '%s'" % split(model_name)[-1])
        embedding_model.save(model_name)
    embedding_weights = [np.array([embedding_model[w] if w in embedding_model else np.random.uniform(-0.25, 0.25, embedding_model.vector_size) for w in vocabulary_inv])]
    return embedding_weights
if __name__ == '__main__':
    import data_helpers
    print('Loading data...')
    (x, _, _, vocabulary_inv) = data_helpers.load_data()
    w = train_word2vec(x, vocabulary_inv)