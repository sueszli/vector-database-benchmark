"""
Title: Text generation with a miniature GPT
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2020/05/29
Last modified: 2020/05/29
Description: Implement a miniature version of GPT and train it to generate text.
Accelerator: GPU
"""
'\n## Introduction\n\nThis example demonstrates how to implement an autoregressive language model\nusing a miniature version of the GPT model.\nThe model consists of a single Transformer block with causal masking\nin its attention layer.\nWe use the text from the IMDB sentiment classification dataset for training\nand generate new movie reviews for a given prompt.\nWhen using this script with your own dataset, make sure it has at least\n1 million words.\n\nThis example should be run with `tf-nightly>=2.3.0-dev20200531` or\nwith TensorFlow 2.3 or higher.\n\n**References:**\n\n- [GPT](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035)\n- [GPT-2](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe)\n- [GPT-3](https://arxiv.org/abs/2005.14165)\n'
'\n## Setup\n'
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras import layers
from keras import ops
from keras.layers import TextVectorization
import numpy as np
import os
import string
import random
import tensorflow
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
'\n## Implement a Transformer block as a layer\n'

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    if False:
        i = 10
        return i + 15
    "\n    Mask the upper half of the dot product matrix in self attention.\n    This prevents flow of information from future tokens to current token.\n    1's in the lower triangle, counting from the lower right corner.\n    "
    i = ops.arange(n_dest)[:, None]
    j = ops.arange(n_src)
    m = i >= j - n_src + n_dest
    mask = ops.cast(m, dtype)
    mask = ops.reshape(mask, [1, n_dest, n_src])
    mult = ops.concatenate([ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])], 0)
    return ops.tile(mask, mult)

class TransformerBlock(layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation='relu'), layers.Dense(embed_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-06)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-06)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, 'bool')
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
'\n## Implement an embedding layer\n\nCreate two separate embedding layers: one for tokens and one for token index\n(positions).\n'

class TokenAndPositionEmbedding(layers.Layer):

    def __init__(self, maxlen, vocab_size, embed_dim):
        if False:
            print('Hello World!')
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        if False:
            while True:
                i = 10
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(0, maxlen, 1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
'\n## Implement the miniature GPT model\n'
vocab_size = 20000
maxlen = 80
embed_dim = 256
num_heads = 2
feed_forward_dim = 256

def create_model():
    if False:
        i = 10
        return i + 15
    inputs = layers.Input(shape=(maxlen,), dtype='int32')
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
    x = transformer_block(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile('adam', loss=[loss_fn, None])
    return model
'\n## Prepare the data for word-level language modelling\n\nDownload the IMDB dataset and combine training and validation sets for a text\ngeneration task.\n'
'shell\ncurl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz\ntar -xf aclImdb_v1.tar.gz\n'
batch_size = 128
filenames = []
directories = ['aclImdb/train/pos', 'aclImdb/train/neg', 'aclImdb/test/pos', 'aclImdb/test/neg']
for dir in directories:
    for f in os.listdir(dir):
        filenames.append(os.path.join(dir, f))
print(f'{len(filenames)} files')
random.shuffle(filenames)
text_ds = tf_data.TextLineDataset(filenames)
text_ds = text_ds.shuffle(buffer_size=256)
text_ds = text_ds.batch(batch_size)

def custom_standardization(input_string):
    if False:
        for i in range(10):
            print('nop')
    'Remove html line-break tags and handle punctuation'
    lowercased = tf_strings.lower(input_string)
    stripped_html = tf_strings.regex_replace(lowercased, '<br />', ' ')
    return tf_strings.regex_replace(stripped_html, f'([{string.punctuation}])', ' \\1')
vectorize_layer = TextVectorization(standardize=custom_standardization, max_tokens=vocab_size - 1, output_mode='int', output_sequence_length=maxlen + 1)
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()

def prepare_lm_inputs_labels(text):
    if False:
        while True:
            i = 10
    '\n    Shift word sequences by 1 position so that the target for position (i) is\n    word at position (i+1). The model will use all words up till position (i)\n    to predict the next word.\n    '
    text = tensorflow.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return (x, y)
text_ds = text_ds.map(prepare_lm_inputs_labels, num_parallel_calls=tf_data.AUTOTUNE)
text_ds = text_ds.prefetch(tf_data.AUTOTUNE)
'\n## Implement a Keras callback for generating text\n'

class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(self, max_tokens, start_tokens, index_to_word, top_k=10, print_every=1):
        if False:
            while True:
                i = 10
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        if False:
            print('Hello World!')
        (logits, indices) = ops.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype('int32')
        preds = keras.activations.softmax(ops.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype('float32')
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        if False:
            return 10
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        if False:
            while True:
                i = 10
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:maxlen]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            (y, _) = self.model.predict(x, verbose=0)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = ' '.join([self.detokenize(_) for _ in self.start_tokens + tokens_generated])
        print(f'generated text:\n{txt}\n')
word_to_index = {}
for (index, word) in enumerate(vocab):
    word_to_index[word] = index
start_prompt = 'this movie is'
start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
num_tokens_generated = 40
text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, vocab)
'\n## Train the model\n\nNote: This code should preferably be run on GPU.\n'
model = create_model()
model.fit(text_ds, verbose=2, epochs=25, callbacks=[text_gen_callback])