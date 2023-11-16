"""
Title: Character-level recurrent sequence-to-sequence model
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2017/09/29
Last modified: 2020/04/26
Description: Character-level recurrent sequence-to-sequence model.
Accelerator: GPU
"""
'\n## Introduction\n\nThis example demonstrates how to implement a basic character-level\nrecurrent sequence-to-sequence model. We apply it to translating\nshort English sentences into short French sentences,\ncharacter-by-character. Note that it is fairly unusual to\ndo character-level machine translation, as word-level\nmodels are more common in this domain.\n\n**Summary of the algorithm**\n\n- We start with input sequences from a domain (e.g. English sentences)\n    and corresponding target sequences from another domain\n    (e.g. French sentences).\n- An encoder LSTM turns input sequences to 2 state vectors\n    (we keep the last LSTM state and discard the outputs).\n- A decoder LSTM is trained to turn the target sequences into\n    the same sequence but offset by one timestep in the future,\n    a training process called "teacher forcing" in this context.\n    It uses as initial state the state vectors from the encoder.\n    Effectively, the decoder learns to generate `targets[t+1...]`\n    given `targets[...t]`, conditioned on the input sequence.\n- In inference mode, when we want to decode unknown input sequences, we:\n    - Encode the input sequence into state vectors\n    - Start with a target sequence of size 1\n        (just the start-of-sequence character)\n    - Feed the state vectors and 1-char target sequence\n        to the decoder to produce predictions for the next character\n    - Sample the next character using these predictions\n        (we simply use argmax).\n    - Append the sampled character to the target sequence\n    - Repeat until we generate the end-of-sequence character or we\n        hit the character limit.\n'
'\n## Setup\n'
import numpy as np
import keras
import os
from pathlib import Path
'\n## Download the data\n'
fpath = keras.utils.get_file(origin='http://www.manythings.org/anki/fra-eng.zip')
dirpath = Path(fpath).parent.absolute()
os.system(f'unzip -q {fpath} -d {dirpath}')
'\n## Configuration\n'
batch_size = 64
epochs = 100
latent_dim = 256
num_samples = 10000
data_path = os.path.join(dirpath, 'fra.txt')
'\n## Prepare the data\n'
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[:min(num_samples, len(lines) - 1)]:
    (input_text, target_text, _) = line.split('\t')
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
input_token_index = dict([(char, i) for (i, char) in enumerate(input_characters)])
target_token_index = dict([(char, i) for (i, char) in enumerate(target_characters)])
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
for (i, (input_text, target_text)) in enumerate(zip(input_texts, target_texts)):
    for (t, char) in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.0
    for (t, char) in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.0
    decoder_target_data[i, t:, target_token_index[' ']] = 1.0
'\n## Build the model\n'
encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
encoder = keras.layers.LSTM(latent_dim, return_state=True)
(encoder_outputs, state_h, state_c) = encoder(encoder_inputs)
encoder_states = [state_h, state_c]
decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
(decoder_outputs, _, _) = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = keras.layers.Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
'\n## Train the model\n'
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.2)
model.save('s2s_model.keras')
'\n## Run inference (sampling)\n\n1. encode input and retrieve initial decoder state\n2. run one step of decoder with this initial state\nand a "start of sequence" token as target.\nOutput will be the next target token.\n3. Repeat with the current target token and current states\n'
model = keras.models.load_model('s2s_model.keras')
encoder_inputs = model.input[0]
(encoder_outputs, state_h_enc, state_c_enc) = model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)
decoder_inputs = model.input[1]
decoder_state_input_h = keras.Input(shape=(latent_dim,))
decoder_state_input_c = keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
(decoder_outputs, state_h_dec, state_c_dec) = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
reverse_input_char_index = dict(((i, char) for (char, i) in input_token_index.items()))
reverse_target_char_index = dict(((i, char) for (char, i) in target_token_index.items()))

def decode_sequence(input_seq):
    if False:
        for i in range(10):
            print('nop')
    states_value = encoder_model.predict(input_seq, verbose=0)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.0
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        (output_tokens, h, c) = decoder_model.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [h, c]
    return decoded_sentence
'\nYou can now generate decoded sentences as such:\n'
for seq_index in range(20):
    input_seq = encoder_input_data[seq_index:seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)