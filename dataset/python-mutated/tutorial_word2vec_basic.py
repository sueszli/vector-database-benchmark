"""Vector Representations of Words.

This is the minimalistic reimplementation of
tensorflow/examples/tutorials/word2vec/word2vec_basic.py
This basic example contains the code needed to download some data,
train on it a bit and visualize the result by using t-SNE.

Once you get comfortable with reading and running the basic version,
you can graduate to
tensorflow/models/embedding/word2vec.py
which is a more serious implementation that showcases some more advanced
TensorFlow principles about how to efficiently use threads to move data
into a text model, how to checkpoint during training, etc.

If your model is no longer I/O bound but you want still more performance, you
can take things further by writing your own TensorFlow Ops, as described in
Adding a New Op. Again we've provided an example of this for the Skip-Gram case
tensorflow/models/embedding/word2vec_optimized.py.

Link
------
https://www.tensorflow.org/versions/r0.9/tutorials/word2vec/index.html#vector-representations-of-words

"""
import argparse
import os
import time
import numpy as np
import tensorflow as tf
from six.moves import xrange
import tensorlayer as tl
import wget
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='one', type=str, required=False, help="The model name. It can be 'one', 'two', 'three', 'four'.")
FLAGS = parser.parse_args()

def main_word2vec_basic():
    if False:
        while True:
            i = 10
    words = tl.files.load_matt_mahoney_text8_dataset()
    data_size = len(words)
    print(data_size)
    print(words[0:10])
    resume = False
    _UNK = '_UNK'
    if FLAGS.model == 'one':
        vocabulary_size = 50000
        batch_size = 128
        embedding_size = 128
        skip_window = 1
        num_skips = 2
        num_sampled = 64
        learning_rate = 1.0
        n_epoch = 20
        model_file_name = 'model_word2vec_50k_128'
    elif FLAGS.model == 'two':
        vocabulary_size = 80000
        batch_size = 20
        embedding_size = 200
        skip_window = 5
        num_skips = 10
        num_sampled = 100
        learning_rate = 0.2
        n_epoch = 15
        model_file_name = 'model_word2vec_80k_200'
    elif FLAGS.model == 'three':
        vocabulary_size = 80000
        batch_size = 500
        embedding_size = 200
        skip_window = 5
        num_skips = 10
        num_sampled = 25
        learning_rate = 0.025
        n_epoch = 20
        model_file_name = 'model_word2vec_80k_200_opt'
    elif FLAGS.model == 'four':
        vocabulary_size = 80000
        batch_size = 100
        embedding_size = 600
        skip_window = 5
        num_skips = 10
        num_sampled = 25
        learning_rate = 0.03
        n_epoch = 200 * 10
        model_file_name = 'model_word2vec_80k_600'
    else:
        raise Exception('Invalid model: %s' % FLAGS.model)
    num_steps = int(data_size / batch_size * n_epoch)
    print('%d Steps in a Epoch, total Epochs %d' % (int(data_size / batch_size), n_epoch))
    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)
    print()
    if resume:
        print('Load existing data and dictionaries' + '!' * 10)
        all_var = tl.files.load_npy_to_any(name=model_file_name + '.npy')
        data = all_var['data']
        count = all_var['count']
        dictionary = all_var['dictionary']
        reverse_dictionary = all_var['reverse_dictionary']
    else:
        (data, count, dictionary, reverse_dictionary) = tl.nlp.build_words_dataset(words, vocabulary_size, True, _UNK)
    print('Most 5 common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
    del words
    print()
    (batch, labels, data_index) = tl.nlp.generate_skip_gram_batch(data=data, batch_size=8, num_skips=4, skip_window=2, data_index=0)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
    (batch, labels, data_index) = tl.nlp.generate_skip_gram_batch(data=data, batch_size=8, num_skips=2, skip_window=1, data_index=0)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
    print()
    valid_size = 16
    valid_window = 100
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    inputs = tl.layers.Input([batch_size], dtype=tf.int32)
    labels = tl.layers.Input([batch_size, 1], dtype=tf.int32)
    emb_net = tl.layers.Word2vecEmbedding(vocabulary_size=vocabulary_size, embedding_size=embedding_size, num_sampled=num_sampled, activate_nce_loss=True, nce_loss_args={}, E_init=tl.initializers.random_uniform(minval=-1.0, maxval=1.0), nce_W_init=tl.initializers.truncated_normal(stddev=float(1.0 / np.sqrt(embedding_size))), nce_b_init=tl.initializers.constant(value=0.0), name='word2vec_layer')
    (emb, nce) = emb_net([inputs, labels])
    model = tl.models.Model(inputs=[inputs, labels], outputs=[emb, nce], name='word2vec_model')
    optimizer = tf.optimizers.Adagrad(learning_rate, initial_accumulator_value=0.1)
    normalized_embeddings = emb_net.normalized_embeddings
    model.train()
    if resume:
        print('Load existing model' + '!' * 10)
        model.load_weights(filepath=model_file_name + '.hdf5')
    tl.nlp.save_vocab(count, name='vocab_text8.txt')
    average_loss = 0
    step = 0
    print_freq = 2000
    while step < num_steps:
        start_time = time.time()
        (batch_inputs, batch_labels, data_index) = tl.nlp.generate_skip_gram_batch(data=data, batch_size=batch_size, num_skips=num_skips, skip_window=skip_window, data_index=data_index)
        with tf.GradientTape() as tape:
            (outputs, nce_cost) = model([batch_inputs, batch_labels])
        grad = tape.gradient(nce_cost, model.trainable_weights)
        optimizer.apply_gradients(zip(grad, model.trainable_weights))
        average_loss += nce_cost
        if step % print_freq == 0:
            if step > 0:
                average_loss /= print_freq
            print('Average loss at step %d/%d. loss: %f took: %fs/per step' % (step, num_steps, average_loss, time.time() - start_time))
            average_loss = 0
        if step % (print_freq * 5) == 0:
            valid_embed = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
            sim = tf.matmul(valid_embed, normalized_embeddings, transpose_b=True)
            sim = sim.numpy()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
        if step % (print_freq * 20) == 0 and step != 0:
            print('Save model, data and dictionaries' + '!' * 10)
            model.save_weights(filepath=model_file_name + '.hdf5')
            tl.files.save_any_to_npy(save_dict={'data': data, 'count': count, 'dictionary': dictionary, 'reverse_dictionary': reverse_dictionary}, name=model_file_name + '.npy')
        step += 1
    print()
    final_embeddings = normalized_embeddings
    tl.visualize.tsne_embedding(final_embeddings, reverse_dictionary, plot_only=500, second=5, saveable=False, name='word2vec_basic')
    print()
    model.eval()
    if not os.path.exists('questions-words.txt'):
        print("Downloading file 'questions-words.txt'")
        wget.download('http://download.tensorflow.org/data/questions-words.txt')
    analogy_questions = tl.nlp.read_analogies_file(eval_file='questions-words.txt', word2id=dictionary)
    n_answer = 4

    def predict(analogy):
        if False:
            for i in range(10):
                print('nop')
        analogy_a = analogy[:, 0]
        analogy_b = analogy[:, 1]
        analogy_c = analogy[:, 2]
        a_emb = tf.gather(normalized_embeddings, analogy_a)
        b_emb = tf.gather(normalized_embeddings, analogy_b)
        c_emb = tf.gather(normalized_embeddings, analogy_c)
        target = c_emb + (b_emb - a_emb)
        dist = tf.matmul(target, normalized_embeddings, transpose_b=True)
        'Predict the top 4 answers for analogy questions.'
        (_, pred_idx) = tf.nn.top_k(dist, n_answer)
        return pred_idx
    correct = 0
    total = analogy_questions.shape[0]
    start = 0
    while start < total:
        limit = start + 2500
        sub = analogy_questions[start:limit, :]
        idx = predict(sub)
        start = limit
        for question in xrange(sub.shape[0]):
            for j in xrange(n_answer):
                if idx[question, j] == sub[question, 3]:
                    print(j + 1, tl.nlp.word_ids_to_words([idx[question, j]], reverse_dictionary), ':', tl.nlp.word_ids_to_words(sub[question, :], reverse_dictionary))
                    correct += 1
                    break
                elif idx[question, j] in sub[question, :3]:
                    continue
                else:
                    break
    print('Eval %4d/%d accuracy = %4.1f%%' % (correct, total, correct * 100.0 / total))
if __name__ == '__main__':
    main_word2vec_basic()