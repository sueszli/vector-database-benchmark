"""
Title: English-to-Spanish translation with a sequence-to-sequence Transformer
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2021/05/26
Last modified: 2023/02/25
Description: Implementing a sequence-to-sequence Transformer and training it on a machine translation task.
Accelerator: GPU
"""
"\n## Introduction\n\nIn this example, we'll build a sequence-to-sequence Transformer model, which\nwe'll train on an English-to-Spanish machine translation task.\n\nYou'll learn how to:\n\n- Vectorize text using the Keras `TextVectorization` layer.\n- Implement a `TransformerEncoder` layer, a `TransformerDecoder` layer,\nand a `PositionalEmbedding` layer.\n- Prepare data for training a sequence-to-sequence model.\n- Use the trained model to generate translations of never-seen-before\ninput sentences (sequence-to-sequence inference).\n\nThe code featured here is adapted from the book\n[Deep Learning with Python, Second Edition](https://www.manning.com/books/deep-learning-with-python-second-edition)\n(chapter 11: Deep learning for text).\nThe present example is fairly barebones, so for detailed explanations of\nhow each building block works, as well as the theory behind Transformers,\nI recommend reading the book.\n"
'\n## Setup\n'
import os
os['KERAS_BACKEND'] = 'tensorflow'
import pathlib
import random
import string
import re
import numpy as np
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
import keras
from keras import layers
from keras import ops
from keras.layers import TextVectorization
"\n## Downloading the data\n\nWe'll be working with an English-to-Spanish translation dataset\nprovided by [Anki](https://www.manythings.org/anki/). Let's download it:\n"
text_file = keras.utils.get_file(fname='spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip', extract=True)
text_file = pathlib.Path(text_file).parent / 'spa-eng' / 'spa.txt'
'\n## Parsing the data\n\nEach line contains an English sentence and its corresponding Spanish sentence.\nThe English sentence is the *source sequence* and Spanish one is the *target sequence*.\nWe prepend the token `"[start]"` and we append the token `"[end]"` to the Spanish sentence.\n'
with open(text_file) as f:
    lines = f.read().split('\n')[:-1]
text_pairs = []
for line in lines:
    (eng, spa) = line.split('\t')
    spa = '[start] ' + spa + ' [end]'
    text_pairs.append((eng, spa))
"\nHere's what our sentence pairs look like:\n"
for _ in range(5):
    print(random.choice(text_pairs))
"\nNow, let's split the sentence pairs into a training set, a validation set,\nand a test set.\n"
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples:]
print(f'{len(text_pairs)} total pairs')
print(f'{len(train_pairs)} training pairs')
print(f'{len(val_pairs)} validation pairs')
print(f'{len(test_pairs)} test pairs')
'\n## Vectorizing the text data\n\nWe\'ll use two instances of the `TextVectorization` layer to vectorize the text\ndata (one for English and one for Spanish),\nthat is to say, to turn the original strings into integer sequences\nwhere each integer represents the index of a word in a vocabulary.\n\nThe English layer will use the default string standardization (strip punctuation characters)\nand splitting scheme (split on whitespace), while\nthe Spanish layer will use a custom standardization, where we add the character\n`"¿"` to the set of punctuation characters to be stripped.\n\nNote: in a production-grade machine translation model, I would not recommend\nstripping the punctuation characters in either language. Instead, I would recommend turning\neach punctuation character into its own token,\nwhich you could achieve by providing a custom `split` function to the `TextVectorization` layer.\n'
strip_chars = string.punctuation + '¿'
strip_chars = strip_chars.replace('[', '')
strip_chars = strip_chars.replace(']', '')
vocab_size = 15000
sequence_length = 20
batch_size = 64

def custom_standardization(input_string):
    if False:
        print('Hello World!')
    lowercase = tf_strings.lower(input_string)
    return tf_strings.regex_replace(lowercase, '[%s]' % re.escape(strip_chars), '')
eng_vectorization = TextVectorization(max_tokens=vocab_size, output_mode='int', output_sequence_length=sequence_length)
spa_vectorization = TextVectorization(max_tokens=vocab_size, output_mode='int', output_sequence_length=sequence_length + 1, standardize=custom_standardization)
train_eng_texts = [pair[0] for pair in train_pairs]
train_spa_texts = [pair[1] for pair in train_pairs]
eng_vectorization.adapt(train_eng_texts)
spa_vectorization.adapt(train_spa_texts)
'\nNext, we\'ll format our datasets.\n\nAt each training step, the model will seek to predict target words N+1 (and beyond)\nusing the source sentence and the target words 0 to N.\n\nAs such, the training dataset will yield a tuple `(inputs, targets)`, where:\n\n- `inputs` is a dictionary with the keys `encoder_inputs` and `decoder_inputs`.\n`encoder_inputs` is the vectorized source sentence and `encoder_inputs` is the target sentence "so far",\nthat is to say, the words 0 to N used to predict word N+1 (and beyond) in the target sentence.\n- `target` is the target sentence offset by one step:\nit provides the next words in the target sentence -- what the model will try to predict.\n'

def format_dataset(eng, spa):
    if False:
        for i in range(10):
            print('nop')
    eng = eng_vectorization(eng)
    spa = spa_vectorization(spa)
    return ({'encoder_inputs': eng, 'decoder_inputs': spa[:, :-1]}, spa[:, 1:])

def make_dataset(pairs):
    if False:
        while True:
            i = 10
    (eng_texts, spa_texts) = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf_data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()
train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)
"\nLet's take a quick look at the sequence shapes\n(we have batches of 64 pairs, and all sequences are 20 steps long):\n"
for (inputs, targets) in train_ds.take(1):
    print(f"""inputs["encoder_inputs"].shape: {inputs['encoder_inputs'].shape}""")
    print(f"""inputs["decoder_inputs"].shape: {inputs['decoder_inputs'].shape}""")
    print(f'targets.shape: {targets.shape}')
'\n## Building the model\n\nOur sequence-to-sequence Transformer consists of a `TransformerEncoder`\nand a `TransformerDecoder` chained together. To make the model aware of word order,\nwe also use a `PositionalEmbedding` layer.\n\nThe source sequence will be pass to the `TransformerEncoder`,\nwhich will produce a new representation of it.\nThis new representation will then be passed\nto the `TransformerDecoder`, together with the target sequence so far (target words 0 to N).\nThe `TransformerDecoder` will then seek to predict the next words in the target sequence (N+1 and beyond).\n\nA key detail that makes this possible is causal masking\n(see method `get_causal_attention_mask()` on the `TransformerDecoder`).\nThe `TransformerDecoder` sees the entire sequences at once, and thus we must make\nsure that it only uses information from target tokens 0 to N when predicting token N+1\n(otherwise, it could use information from the future, which would\nresult in a model that cannot be used at inference time).\n'
import keras.ops as ops

class TransformerEncoder(layers.Layer):

    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation='relu'), layers.Dense(embed_dim)])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if False:
            for i in range(10):
                print('nop')
        if mask is not None:
            padding_mask = ops.cast(mask[:, None, :], dtype='int32')
        else:
            padding_mask = None
        attention_output = self.attention(query=inputs, value=inputs, key=inputs, attention_mask=padding_mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        if False:
            i = 10
            return i + 15
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim, 'dense_dim': self.dense_dim, 'num_heads': self.num_heads})
        return config

class PositionalEmbedding(layers.Layer):

    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        length = ops.shape(inputs)[-1]
        positions = ops.arange(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        if False:
            for i in range(10):
                print('nop')
        if mask is None:
            return None
        else:
            return ops.not_equal(inputs, 0)

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        config = super().get_config()
        config.update({'sequence_length': self.sequence_length, 'vocab_size': self.vocab_size, 'embed_dim': self.embed_dim})
        return config

class TransformerDecoder(layers.Layer):

    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([layers.Dense(latent_dim, activation='relu'), layers.Dense(embed_dim)])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        if False:
            print('Hello World!')
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = ops.cast(mask[:, None, :], dtype='int32')
            padding_mask = ops.minimum(padding_mask, causal_mask)
        else:
            padding_mask = None
        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=causal_mask)
        out_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(query=out_1, value=encoder_outputs, key=encoder_outputs, attention_mask=padding_mask)
        out_2 = self.layernorm_2(out_1 + attention_output_2)
        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        if False:
            return 10
        input_shape = ops.shape(inputs)
        (batch_size, sequence_length) = (input_shape[0], input_shape[1])
        i = ops.arange(sequence_length)[:, None]
        j = ops.arange(sequence_length)
        mask = ops.cast(i >= j, dtype='int32')
        mask = ops.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = ops.concatenate([ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])], axis=0)
        return ops.tile(mask, mult)

    def get_config(self):
        if False:
            print('Hello World!')
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim, 'latent_dim': self.latent_dim, 'num_heads': self.num_heads})
        return config
'\nNext, we assemble the end-to-end model.\n'
embed_dim = 256
latent_dim = 2048
num_heads = 8
encoder_inputs = keras.Input(shape=(None,), dtype='int64', name='encoder_inputs')
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
encoder = keras.Model(encoder_inputs, encoder_outputs)
decoder_inputs = keras.Input(shape=(None,), dtype='int64', name='decoder_inputs')
encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name='decoder_state_inputs')
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation='softmax')(x)
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)
decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name='transformer')
"\n## Training our model\n\nWe'll use accuracy as a quick way to monitor training progress on the validation data.\nNote that machine translation typically uses BLEU scores as well as other metrics, rather than accuracy.\n\nHere we only train for 1 epoch, but to get the model to actually converge\nyou should train for at least 30 epochs.\n"
epochs = 1
transformer.summary()
transformer.compile('rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
transformer.fit(train_ds, epochs=epochs, validation_data=val_ds)
'\n## Decoding test sentences\n\nFinally, let\'s demonstrate how to translate brand new English sentences.\nWe simply feed into the model the vectorized English sentence\nas well as the target token `"[start]"`, then we repeatedly generated the next token, until\nwe hit the token `"[end]"`.\n'
spa_vocab = spa_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20

def decode_sequence(input_sentence):
    if False:
        for i in range(10):
            print('nop')
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = '[start]'
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = spa_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = ops.convert_to_numpy(ops.argmax(predictions[0, i, :])).item(0)
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += ' ' + sampled_token
        if sampled_token == '[end]':
            break
    return decoded_sentence
test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(30):
    input_sentence = random.choice(test_eng_texts)
    translated = decode_sequence(input_sentence)
'\nAfter 30 epochs, we get results such as:\n\n> She handed him the money.\n> [start] ella le pasó el dinero [end]\n\n> Tom has never heard Mary sing.\n> [start] tom nunca ha oído cantar a mary [end]\n\n> Perhaps she will come tomorrow.\n> [start] tal vez ella vendrá mañana [end]\n\n> I love to write.\n> [start] me encanta escribir [end]\n\n> His French is improving little by little.\n> [start] su francés va a [UNK] sólo un poco [end]\n\n> My hotel told me to call you.\n> [start] mi hotel me dijo que te [UNK] [end]\n'