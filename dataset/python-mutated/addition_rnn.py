"""
Title: Sequence to sequence learning for performing number addition
Author: [Smerity](https://twitter.com/Smerity) and others
Date created: 2015/08/17
Last modified: 2020/04/17
Description: A model that learns to add strings of numbers, e.g. "535+61" -> "596".
Accelerator: GPU
"""
'\n## Introduction\n\nIn this example, we train a model to learn to add two numbers, provided as strings.\n\n**Example:**\n\n- Input: "535+61"\n- Output: "596"\n\nInput may optionally be reversed, which was shown to increase performance in many tasks\n in: [Learning to Execute](http://arxiv.org/abs/1410.4615) and\n[Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf).\n\nTheoretically, sequence order inversion introduces shorter term dependencies between\n source and target for this problem.\n\n**Results:**\n\nFor two digits (reversed):\n\n+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs\n\nThree digits (reversed):\n\n+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs\n\nFour digits (reversed):\n\n+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs\n\nFive digits (reversed):\n\n+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs\n'
'\n## Setup\n'
import keras
from keras import layers
import numpy as np
TRAINING_SIZE = 50000
DIGITS = 3
REVERSE = True
MAXLEN = DIGITS + 1 + DIGITS
'\n## Generate the data\n'

class CharacterTable:
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot or integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(self, chars):
        if False:
            return 10
        'Initialize character table.\n        # Arguments\n            chars: Characters that can appear in the input.\n        '
        self.chars = sorted(set(chars))
        self.char_indices = dict(((c, i) for (i, c) in enumerate(self.chars)))
        self.indices_char = dict(((i, c) for (i, c) in enumerate(self.chars)))

    def encode(self, C, num_rows):
        if False:
            print('Hello World!')
        'One-hot encode given string C.\n        # Arguments\n            C: string, to be encoded.\n            num_rows: Number of rows in the returned one-hot encoding. This is\n                used to keep the # of rows for each data the same.\n        '
        x = np.zeros((num_rows, len(self.chars)))
        for (i, c) in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if False:
            for i in range(10):
                print('nop')
        'Decode the given vector or 2D array to their character output.\n        # Arguments\n            x: A vector or a 2D array of probabilities or one-hot representations;\n                or a vector of character indices (used with `calc_argmax=False`).\n            calc_argmax: Whether to find the character index with maximum\n                probability, defaults to `True`.\n        '
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join((self.indices_char[x] for x in x))
chars = '0123456789+ '
ctable = CharacterTable(chars)
questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    f = lambda : int(''.join((np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1)))))
    (a, b) = (f(), f())
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    ans += ' ' * (DIGITS + 1 - len(ans))
    if REVERSE:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total questions:', len(questions))
'\n## Vectorize the data\n'
print('Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=bool)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=bool)
for (i, sentence) in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for (i, sentence) in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]
split_at = len(x) - len(x) // 10
(x_train, x_val) = (x[:split_at], x[split_at:])
(y_train, y_val) = (y[:split_at], y[split_at:])
print('Training Data:')
print(x_train.shape)
print(y_train.shape)
print('Validation Data:')
print(x_val.shape)
print(y_val.shape)
'\n## Build the model\n'
print('Build model...')
num_layers = 1
model = keras.Sequential()
model.add(layers.Input((MAXLEN, len(chars))))
model.add(layers.LSTM(128))
model.add(layers.RepeatVector(DIGITS + 1))
for _ in range(num_layers):
    model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.Dense(len(chars), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
'\n## Train the model\n'
epochs = 30
batch_size = 32
for epoch in range(1, epochs):
    print()
    print('Iteration', epoch)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_val, y_val))
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        (rowx, rowy) = (x_val[np.array([ind])], y_val[np.array([ind])])
        preds = np.argmax(model.predict(rowx), axis=-1)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print('☑ ' + guess)
        else:
            print('☒ ' + guess)
"\nYou'll get to 99+% validation accuracy after ~30 epochs.\n\nExample available on HuggingFace.\n\n| Trained Model | Demo |\n| :--: | :--: |\n| [![Generic badge](https://img.shields.io/badge/🤗%20Model-Addition%20LSTM-black.svg)](https://huggingface.co/keras-io/addition-lstm) | [![Generic badge](https://img.shields.io/badge/🤗%20Spaces-Addition%20LSTM-black.svg)](https://huggingface.co/spaces/keras-io/addition-lstm) |\n"