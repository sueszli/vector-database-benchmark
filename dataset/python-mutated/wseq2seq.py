from __future__ import print_function, division
from builtins import range, input
import os, sys
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
try:
    import keras.backend as K
    if len(K.tensorflow_backend._get_available_gpus()) > 0:
        from keras.layers import CuDNNLSTM as LSTM
        from keras.layers import CuDNNGRU as GRU
except:
    pass
BATCH_SIZE = 64
EPOCHS = 40
LATENT_DIM = 256
NUM_SAMPLES = 10000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
input_texts = []
target_texts = []
target_texts_inputs = []
t = 0
for line in open('../large_files/translation/spa.txt'):
    t += 1
    if t > NUM_SAMPLES:
        break
    if '\t' not in line:
        continue
    (input_text, translation, *rest) = line.rstrip().split('\t')
    target_text = translation + ' <eos>'
    target_text_input = '<sos> ' + translation
    input_texts.append(input_text)
    target_texts.append(target_text)
    target_texts_inputs.append(target_text_input)
print('num samples:', len(input_texts))
tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
word2idx_inputs = tokenizer_inputs.word_index
print('Found %s unique input tokens.' % len(word2idx_inputs))
max_len_input = max((len(s) for s in input_sequences))
tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)
word2idx_outputs = tokenizer_outputs.word_index
print('Found %s unique output tokens.' % len(word2idx_outputs))
num_words_output = len(word2idx_outputs) + 1
max_len_target = max((len(s) for s in target_sequences))
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
print('encoder_inputs.shape:', encoder_inputs.shape)
print('encoder_inputs[0]:', encoder_inputs[0])
decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
print('decoder_inputs[0]:', decoder_inputs[0])
print('decoder_inputs.shape:', decoder_inputs.shape)
decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')
print('Loading word vectors...')
word2vec = {}
with open(os.path.join('../large_files/glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM)) as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))
print('Filling pre-trained embeddings...')
num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for (word, i) in word2idx_inputs.items():
    if i < MAX_NUM_WORDS:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
embedding_layer = Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=max_len_input)
decoder_targets_one_hot = np.zeros((len(input_texts), max_len_target, num_words_output), dtype='float32')
for (i, d) in enumerate(decoder_targets):
    for (t, word) in enumerate(d):
        if word != 0:
            decoder_targets_one_hot[i, t, word] = 1
encoder_inputs_placeholder = Input(shape=(max_len_input,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(LATENT_DIM, return_state=True)
(encoder_outputs, h, c) = encoder(x)
encoder_states = [h, c]
decoder_inputs_placeholder = Input(shape=(max_len_target,))
decoder_embedding = Embedding(num_words_output, EMBEDDING_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
(decoder_outputs, _, _) = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)
decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)

def custom_loss(y_true, y_pred):
    if False:
        for i in range(10):
            print('nop')
    mask = K.cast(y_true > 0, dtype='float32')
    out = mask * y_true * K.log(y_pred)
    return -K.sum(out) / K.sum(mask)

def acc(y_true, y_pred):
    if False:
        for i in range(10):
            print('nop')
    targ = K.argmax(y_true, axis=-1)
    pred = K.argmax(y_pred, axis=-1)
    correct = K.cast(K.equal(targ, pred), dtype='float32')
    mask = K.cast(K.greater(targ, 0), dtype='float32')
    n_correct = K.sum(mask * correct)
    n_total = K.sum(mask)
    return n_correct / n_total
model.compile(optimizer='adam', loss=custom_loss, metrics=[acc])
r = model.fit([encoder_inputs, decoder_inputs], decoder_targets_one_hot, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
model.save('s2s.h5')
encoder_model = Model(encoder_inputs_placeholder, encoder_states)
decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)
(decoder_outputs, h, c) = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)
decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs_single] + decoder_states_inputs, [decoder_outputs] + decoder_states)
idx2word_eng = {v: k for (k, v) in word2idx_inputs.items()}
idx2word_trans = {v: k for (k, v) in word2idx_outputs.items()}

def decode_sequence(input_seq):
    if False:
        return 10
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']
    output_sentence = []
    for _ in range(max_len_target):
        (output_tokens, h, c) = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])
        if eos == idx:
            break
        word = ''
        if idx > 0:
            word = idx2word_trans[idx]
            output_sentence.append(word)
        target_seq[0, 0] = idx
        states_value = [h, c]
    return ' '.join(output_sentence)
while True:
    i = np.random.choice(len(input_texts))
    input_seq = encoder_inputs[i:i + 1]
    translation = decode_sequence(input_seq)
    print('-')
    print('Input:', input_texts[i])
    print('Translation:', translation)
    ans = input('Continue? [Y/n]')
    if ans and ans.lower().startswith('n'):
        break