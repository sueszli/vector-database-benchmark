"""
Title: End-to-end Masked Language Modeling with BERT
Author: [Ankur Singh](https://twitter.com/ankur310794)
Converted to Keras-Core: [Mrutyunjay Biswal](https://twitter.com/LearnStochastic)
Date created: 2020/09/18
Last modified: 2023/09/06
Description: Implement a Masked Language Model (MLM) with BERT and fine-tune it on the IMDB Reviews dataset.
Accelerator: GPU
"""
'\n## Introduction\n\nMasked Language Modeling is a fill-in-the-blank task,\nwhere a model uses the context words surrounding a mask token to try to predict what the\nmasked word should be.\n\nFor an input that contains one or more mask tokens,\nthe model will generate the most likely substitution for each.\n\nExample:\n\n- Input: "I have watched this [MASK] and it was awesome."\n- Output: "I have watched this movie and it was awesome."\n\nMasked language modeling is a great way to train a language\nmodel in a self-supervised setting (without human-annotated labels).\nSuch a model can then be fine-tuned to accomplish various supervised\nNLP tasks.\n\nThis example teaches you how to build a BERT model from scratch,\ntrain it with the masked language modeling task,\nand then fine-tune this model on a sentiment classification task.\n\nWe will use the Keras-Core `TextVectorization` and `MultiHeadAttention` layers\nto create a BERT Transformer-Encoder network architecture.\n\nNote: This is only tensorflow backend compatible.\n'
'\n## Setup\n'
import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import tensorflow as tf
import keras
from keras import layers
'\n## Configuration\n'

@dataclass
class Config:
    MAX_LEN = 256
    BATCH_SIZE = 32
    LR = 0.001
    VOCAB_SIZE = 30000
    EMBED_DIM = 128
    NUM_HEAD = 8
    FF_DIM = 128
    NUM_LAYERS = 1
    NUM_EPOCHS = 1
    STEPS_PER_EPOCH = 2
config = Config()
'\n## Download the Data: IMDB Movie Review Sentiment Classification\n\nDownload the IMDB data and load into a Pandas DataFrame.\n'
fpath = keras.utils.get_file(origin='https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
dirpath = Path(fpath).parent.absolute()
os.system(f'tar -xf {fpath} -C {dirpath}')
'\nThe `aclImdb` folder contains a `train` and `test` subfolder:\n'
os.system(f'ls {dirpath}/aclImdb')
os.system(f'ls {dirpath}/aclImdb/train')
os.system(f'ls {dirpath}/aclImdb/test')
"\nWe are only interested in the `pos` and `neg` subfolders, so let's delete the rest:\n"
os.system(f'rm -r {dirpath}/aclImdb/train/unsup')
os.system(f'rm -r {dirpath}/aclImdb/train/*.feat')
os.system(f'rm -r {dirpath}/aclImdb/train/*.txt')
os.system(f'rm -r {dirpath}/aclImdb/test/*.feat')
os.system(f'rm -r {dirpath}/aclImdb/test/*.txt')
"\nLet's read the dataset from the text files to a DataFrame.\n"

def get_text_list_from_files(files):
    if False:
        print('Hello World!')
    text_list = []
    for name in files:
        with open(name) as f:
            for line in f:
                text_list.append(line)
    return text_list

def get_data_from_text_files(folder_name):
    if False:
        for i in range(10):
            print('nop')
    pos_files = glob.glob(f'{dirpath}/aclImdb/' + folder_name + '/pos/*.txt')
    pos_texts = get_text_list_from_files(pos_files)
    neg_files = glob.glob(f'{dirpath}/aclImdb/' + folder_name + '/neg/*.txt')
    neg_texts = get_text_list_from_files(neg_files)
    df = pd.DataFrame({'review': pos_texts + neg_texts, 'sentiment': [0] * len(pos_texts) + [1] * len(neg_texts)})
    df = df.sample(len(df)).reset_index(drop=True)
    return df
train_df = get_data_from_text_files('train')
test_df = get_data_from_text_files('test')
all_data = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
assert len(all_data) != 0, f'{all_data} is empty'
'\n## Dataset preparation\n\nWe will use the `TextVectorization` layer to vectorize the text into integer token ids.\nIt transforms a batch of strings into either\na sequence of token indices (one sample = 1D array of integer token indices, in order)\nor a dense representation (one sample = 1D array of float values encoding an unordered set of tokens).\n\nBelow, we define 3 preprocessing functions.\n\n1.  The `get_vectorize_layer` function builds the `TextVectorization` layer.\n2.  The `encode` function encodes raw text into integer token ids.\n3.  The `get_masked_input_and_labels` function will mask input token ids. It masks 15% of all input tokens in each sequence at random.\n'

def custom_standardization(input_data):
    if False:
        i = 10
        return i + 15
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape("!#$%&'()*+,-./:;<=>?@\\^_`{|}~"), '')

def get_vectorize_layer(texts, vocab_size, max_seq, special_tokens=['[MASK]']):
    if False:
        print('Hello World!')
    "Build Text vectorization layer\n\n    Args:\n      texts (list): List of string i.e input texts\n      vocab_size (int): vocab size\n      max_seq (int): Maximum sequence lenght.\n      special_tokens (list, optional): List of special tokens. Defaults to `['[MASK]']`.\n\n    Returns:\n        layers.Layer: Return TextVectorization Keras Layer\n    "
    vectorize_layer = layers.TextVectorization(max_tokens=vocab_size, output_mode='int', standardize=custom_standardization, output_sequence_length=max_seq)
    vectorize_layer.adapt(texts)
    vocab = vectorize_layer.get_vocabulary()
    vocab = vocab[2:vocab_size - len(special_tokens)] + ['[mask]']
    vectorize_layer.set_vocabulary(vocab)
    return vectorize_layer
vectorize_layer = get_vectorize_layer(all_data.review.values.tolist(), config.VOCAB_SIZE, config.MAX_LEN, special_tokens=['[mask]'])
mask_token_id = vectorize_layer(['[mask]']).numpy()[0][0]

def encode(texts):
    if False:
        i = 10
        return i + 15
    encoded_texts = vectorize_layer(texts)
    return encoded_texts.numpy()

def get_masked_input_and_labels(encoded_texts):
    if False:
        i = 10
        return i + 15
    inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
    inp_mask[encoded_texts <= 2] = False
    labels = -1 * np.ones(encoded_texts.shape, dtype=int)
    labels[inp_mask] = encoded_texts[inp_mask]
    encoded_texts_masked = np.copy(encoded_texts)
    inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < 0.9)
    encoded_texts_masked[inp_mask_2mask] = mask_token_id
    inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < 1 / 9)
    encoded_texts_masked[inp_mask_2random] = np.random.randint(3, mask_token_id, inp_mask_2random.sum())
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0
    y_labels = np.copy(encoded_texts)
    return (encoded_texts_masked, y_labels, sample_weights)
x_train = encode(train_df.review.values)
y_train = train_df.sentiment.values
train_classifier_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(config.BATCH_SIZE)
x_test = encode(test_df.review.values)
y_test = test_df.sentiment.values
test_classifier_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(config.BATCH_SIZE)
test_raw_classifier_ds = tf.data.Dataset.from_tensor_slices((test_df.review.values, y_test)).batch(config.BATCH_SIZE)
x_all_review = encode(all_data.review.values)
(x_masked_train, y_masked_labels, sample_weights) = get_masked_input_and_labels(x_all_review)
mlm_ds = tf.data.Dataset.from_tensor_slices((x_masked_train, y_masked_labels, sample_weights))
mlm_ds = mlm_ds.shuffle(1000).batch(config.BATCH_SIZE)
id2token = dict(enumerate(vectorize_layer.get_vocabulary()))
token2id = {y: x for (x, y) in id2token.items()}

class MaskedTextGenerator(keras.callbacks.Callback):

    def __init__(self, sample_tokens, top_k=5):
        if False:
            i = 10
            return i + 15
        self.sample_tokens = sample_tokens
        self.k = top_k

    def decode(self, tokens):
        if False:
            print('Hello World!')
        return ' '.join([id2token[t] for t in tokens if t != 0])

    def convert_ids_to_tokens(self, id):
        if False:
            print('Hello World!')
        return id2token[id]

    def on_epoch_end(self, epoch, logs=None):
        if False:
            while True:
                i = 10
        prediction = self.model.predict(self.sample_tokens)
        masked_index = np.where(self.sample_tokens == mask_token_id)
        masked_index = masked_index[1]
        mask_prediction = prediction[0][masked_index]
        top_indices = mask_prediction[0].argsort()[-self.k:][::-1]
        values = mask_prediction[0][top_indices]
        for i in range(len(top_indices)):
            p = top_indices[i]
            v = values[i]
            tokens = np.copy(self.sample_tokens[0])
            tokens[masked_index[0]] = p
            result = {'input_text': self.decode(self.sample_tokens[0]), 'prediction': self.decode(tokens), 'probability': v, 'predicted mask token': self.convert_ids_to_tokens(p)}
sample_tokens = vectorize_layer(['I have watched this [mask] and it was awesome'])
generator_callback = MaskedTextGenerator(sample_tokens.numpy())
'\n## Create BERT model (Pretraining Model) for masked language modeling\n\nWe will create a BERT-like pretraining model architecture\nusing the `MultiHeadAttention` layer.\nIt will take token ids as inputs (including masked tokens)\nand it will predict the correct ids for the masked input tokens.\n'

def bert_module(query, key, value, layer_num):
    if False:
        print('Hello World!')
    attention_output = layers.MultiHeadAttention(num_heads=config.NUM_HEAD, key_dim=config.EMBED_DIM // config.NUM_HEAD, name=f'encoder_{layer_num}_multiheadattention')(query, key, value)
    attention_output = layers.Dropout(0.1, name=f'encoder_{layer_num}_att_dropout')(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-06, name=f'encoder_{layer_num}_att_layernormalization')(query + attention_output)
    ffn = keras.Sequential([layers.Dense(config.FF_DIM, activation='relu'), layers.Dense(config.EMBED_DIM)], name=f'encoder_{layer_num}_ffn')
    ffn_output = ffn(attention_output)
    ffn_output = layers.Dropout(0.1, name=f'encoder_{layer_num}_ffn_dropout')(ffn_output)
    sequence_output = layers.LayerNormalization(epsilon=1e-06, name=f'encoder_{layer_num}_ffn_layernormalization')(attention_output + ffn_output)
    return sequence_output

def get_pos_encoding_matrix(max_len, d_emb):
    if False:
        i = 10
        return i + 15
    pos_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] if pos != 0 else np.zeros(d_emb) for pos in range(max_len)])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])
    return pos_enc
loss_fn = keras.losses.SparseCategoricalCrossentropy(reduction=None)
loss_tracker = keras.metrics.Mean(name='loss')

class MaskedLanguageModel(keras.Model):

    def train_step(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        if len(inputs) == 3:
            (features, labels, sample_weight) = inputs
        else:
            (features, labels) = inputs
            sample_weight = None
        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            loss = loss_fn(labels, predictions, sample_weight=sample_weight)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        loss_tracker.update_state(loss, sample_weight=sample_weight)
        return {'loss': loss_tracker.result()}

    @property
    def metrics(self):
        if False:
            return 10
        return [loss_tracker]

def create_masked_language_bert_model():
    if False:
        for i in range(10):
            print('nop')
    inputs = layers.Input((config.MAX_LEN,), dtype=tf.int64)
    word_embeddings = layers.Embedding(config.VOCAB_SIZE, config.EMBED_DIM, name='word_embedding')(inputs)
    position_embeddings = layers.Embedding(input_dim=config.MAX_LEN, output_dim=config.EMBED_DIM, embeddings_initializer=keras.initializers.Constant(get_pos_encoding_matrix(config.MAX_LEN, config.EMBED_DIM)), name='position_embedding')(tf.range(start=0, limit=config.MAX_LEN, delta=1))
    embeddings = word_embeddings + position_embeddings
    encoder_output = embeddings
    for i in range(config.NUM_LAYERS):
        encoder_output = bert_module(encoder_output, encoder_output, encoder_output, i)
    mlm_output = layers.Dense(config.VOCAB_SIZE, name='mlm_cls', activation='softmax')(encoder_output)
    mlm_model = MaskedLanguageModel(inputs, mlm_output, name='masked_bert_model')
    optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    mlm_model.compile(optimizer=optimizer)
    return mlm_model
bert_masked_model = create_masked_language_bert_model()
bert_masked_model.summary()
'\n## Train and Save\n'
bert_masked_model.fit(mlm_ds, epochs=Config.NUM_EPOCHS, steps_per_epoch=Config.STEPS_PER_EPOCH, callbacks=[generator_callback])
bert_masked_model.save('bert_mlm_imdb.keras')
"\n## Fine-tune a sentiment classification model\n\nWe will fine-tune our self-supervised model on a downstream task of sentiment classification.\nTo do this, let's create a classifier by adding a pooling layer and a `Dense` layer on top of the\npretrained BERT features.\n"
mlm_model = keras.models.load_model('bert_mlm_imdb.keras', custom_objects={'MaskedLanguageModel': MaskedLanguageModel})
pretrained_bert_model = keras.Model(mlm_model.input, mlm_model.get_layer('encoder_0_ffn_layernormalization').output)
pretrained_bert_model.trainable = False

def create_classifier_bert_model():
    if False:
        return 10
    inputs = layers.Input((config.MAX_LEN,), dtype=tf.int64)
    sequence_output = pretrained_bert_model(inputs)
    pooled_output = layers.GlobalMaxPooling1D()(sequence_output)
    hidden_layer = layers.Dense(64, activation='relu')(pooled_output)
    outputs = layers.Dense(1, activation='sigmoid')(hidden_layer)
    classifer_model = keras.Model(inputs, outputs, name='classification')
    optimizer = keras.optimizers.Adam()
    classifer_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifer_model
classifer_model = create_classifier_bert_model()
classifer_model.summary()
classifer_model.fit(train_classifier_ds, epochs=Config.NUM_EPOCHS, steps_per_epoch=Config.STEPS_PER_EPOCH, validation_data=test_classifier_ds)
pretrained_bert_model.trainable = True
optimizer = keras.optimizers.Adam()
classifer_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
classifer_model.fit(train_classifier_ds, epochs=Config.NUM_EPOCHS, steps_per_epoch=Config.STEPS_PER_EPOCH, validation_data=test_classifier_ds)
"\n## Create an end-to-end model and evaluate it\n\nWhen you want to deploy a model, it's best if it already includes its preprocessing\npipeline, so that you don't have to reimplement the preprocessing logic in your\nproduction environment. Let's create an end-to-end model that incorporates\nthe `TextVectorization` layer, and let's evaluate. Our model will accept raw strings\nas input.\n"

def get_end_to_end(model):
    if False:
        while True:
            i = 10
    inputs_string = keras.Input(shape=(1,), dtype='string')
    indices = vectorize_layer(inputs_string)
    outputs = model(indices)
    end_to_end_model = keras.Model(inputs_string, outputs, name='end_to_end_model')
    optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    end_to_end_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return end_to_end_model
end_to_end_classification_model = get_end_to_end(classifer_model)
end_to_end_classification_model.evaluate(test_raw_classifier_ds)