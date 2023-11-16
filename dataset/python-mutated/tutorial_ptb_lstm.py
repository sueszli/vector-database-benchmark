"""Example of Synced sequence input and output.

This is a reimpmentation of the TensorFlow official PTB example in :
tensorflow/models/rnn/ptb

The batch_size can be seem as how many concurrent computations.\\n
As the following example shows, the first batch learn the sequence information by using 0 to 9.\\n
The second batch learn the sequence information by using 10 to 19.\\n
So it ignores the information from 9 to 10 !\\n
If only if we set the batch_size = 1, it will consider all information from 0 to 20.\\n

The meaning of batch_size here is not the same with the MNIST example. In MNIST example,
batch_size reflects how many examples we consider in each iteration, while in
PTB example, batch_size is how many concurrent processes (segments)
for speed up computation.

Some Information will be ignored if batch_size > 1, however, if your dataset
is "long" enough (a text corpus usually has billions words), the ignored
information would not effect the final result.

In PTB tutorial, we setted batch_size = 20, so we cut the dataset into 20 segments.
At the begining of each epoch, we initialize (reset) the 20 RNN states for 20
segments, then go through 20 segments separately.

The training data will be generated as follow:\\n

>>> train_data = [i for i in range(20)]
>>> for batch in tl.iterate.ptb_iterator(train_data, batch_size=2, num_steps=3):
>>>     x, y = batch
>>>     print(x, '\\n',y)
... [[ 0  1  2] <---x                       1st subset/ iteration
...  [10 11 12]]
... [[ 1  2  3] <---y
...  [11 12 13]]
...
... [[ 3  4  5]  <--- 1st batch input       2nd subset/ iteration
...  [13 14 15]] <--- 2nd batch input
... [[ 4  5  6]  <--- 1st batch target
...  [14 15 16]] <--- 2nd batch target
...
... [[ 6  7  8]                             3rd subset/ iteration
...  [16 17 18]]
... [[ 7  8  9]
...  [17 18 19]]

Hao Dong: This example can also be considered as pre-training of the word
embedding matrix.

About RNN
----------
$ Karpathy Blog : http://karpathy.github.io/2015/05/21/rnn-effectiveness/

More TensorFlow official RNN examples can be found here
---------------------------------------------------------
$ RNN for PTB : https://www.tensorflow.org/versions/master/tutorials/recurrent/index.html#recurrent-neural-networks
$ Seq2seq : https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html#sequence-to-sequence-models
$ translation : tensorflow/models/rnn/translate

Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz


A) use the zero_state function on the cell object

B) for an rnn, all time steps share weights. We use one matrix to keep all
gate weights. Split by column into 4 parts to get the 4 gate weight matrices.

"""
import argparse
import sys
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.models import Model
tl.logging.set_verbosity(tl.logging.DEBUG)

def process_args(args):
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='small', choices=['small', 'medium', 'large'], help='A type of model. Possible options are: small, medium, large.')
    parameters = parser.parse_args(args)
    return parameters

class PTB_Net(Model):

    def __init__(self, vocab_size, hidden_size, init, keep):
        if False:
            return 10
        super(PTB_Net, self).__init__()
        self.embedding = tl.layers.Embedding(vocab_size, hidden_size, init)
        self.dropout1 = tl.layers.Dropout(keep=keep)
        self.lstm1 = tl.layers.RNN(cell=tf.keras.layers.LSTMCell(hidden_size), return_last_output=False, return_last_state=True, return_seq_2d=False, in_channels=hidden_size)
        self.dropout2 = tl.layers.Dropout(keep=keep)
        self.lstm2 = tl.layers.RNN(cell=tf.keras.layers.LSTMCell(hidden_size), return_last_output=False, return_last_state=True, return_seq_2d=True, in_channels=hidden_size)
        self.dropout3 = tl.layers.Dropout(keep=keep)
        self.out_dense = tl.layers.Dense(vocab_size, in_channels=hidden_size, W_init=init, b_init=init, act=None)

    def forward(self, inputs, lstm1_initial_state=None, lstm2_initial_state=None):
        if False:
            return 10
        inputs = self.embedding(inputs)
        inputs = self.dropout1(inputs)
        (lstm1_out, lstm1_state) = self.lstm1(inputs, initial_state=lstm1_initial_state)
        inputs = self.dropout2(lstm1_out)
        (lstm2_out, lstm2_state) = self.lstm2(inputs, initial_state=lstm2_initial_state)
        inputs = self.dropout3(lstm2_out)
        logits = self.out_dense(inputs)
        return (logits, lstm1_state, lstm2_state)

def main():
    if False:
        i = 10
        return i + 15
    '\n    The core of the model consists of an LSTM cell that processes one word at\n    a time and computes probabilities of the possible continuations of the\n    sentence. The memory state of the network is initialized with a vector\n    of zeros and gets updated after reading each word. Also, for computational\n    reasons, we will process data in mini-batches of size batch_size.\n\n    '
    param = process_args(sys.argv[1:])
    if param.model == 'small':
        init_scale = 0.1
        learning_rate = 0.001
        max_grad_norm = 5
        num_steps = 20
        hidden_size = 200
        max_epoch = 4
        max_max_epoch = 13
        keep_prob = 1.0
        lr_decay = 0.5
        batch_size = 20
        vocab_size = 10000
    elif param.model == 'medium':
        init_scale = 0.05
        learning_rate = 0.001
        max_grad_norm = 5
        num_steps = 35
        hidden_size = 650
        max_epoch = 6
        max_max_epoch = 39
        keep_prob = 0.5
        lr_decay = 0.8
        batch_size = 20
        vocab_size = 10000
    elif param.model == 'large':
        init_scale = 0.04
        learning_rate = 0.001
        max_grad_norm = 10
        num_steps = 35
        hidden_size = 1500
        max_epoch = 14
        max_max_epoch = 55
        keep_prob = 0.35
        lr_decay = 1 / 1.15
        batch_size = 20
        vocab_size = 10000
    else:
        raise ValueError('Invalid model: %s', param.model)
    (train_data, valid_data, test_data, vocab_size) = tl.files.load_ptb_dataset()
    print('len(train_data) {}'.format(len(train_data)))
    print('len(valid_data) {}'.format(len(valid_data)))
    print('len(test_data)  {}'.format(len(test_data)))
    print('vocab_size      {}'.format(vocab_size))
    init = tf.random_uniform_initializer(-init_scale, init_scale)
    net = PTB_Net(hidden_size=hidden_size, vocab_size=vocab_size, init=init, keep=keep_prob)
    lr = tf.Variable(0.0, trainable=False)
    train_weights = net.weights
    optimizer = tf.optimizers.Adam(lr=lr)
    print(net)
    print('\nStart learning a language model by using PTB dataset')
    for i in range(max_max_epoch):
        new_lr_decay = lr_decay ** max(i - max_epoch, 0.0)
        lr.assign(learning_rate * new_lr_decay)
        net.train()
        print('Epoch: %d/%d Learning rate: %.3f' % (i + 1, max_max_epoch, lr.value()))
        epoch_size = (len(train_data) // batch_size - 1) // num_steps
        start_time = time.time()
        costs = 0.0
        iters = 0
        lstm1_state = None
        lstm2_state = None
        for (step, (x, y)) in enumerate(tl.iterate.ptb_iterator(train_data, batch_size, num_steps)):
            with tf.GradientTape() as tape:
                (logits, lstm1_state, lstm2_state) = net(x, lstm1_initial_state=lstm1_state, lstm2_initial_state=lstm2_state)
                cost = tl.cost.cross_entropy(logits, tf.reshape(y, [-1]), name='train_loss')
            (grad, _) = tf.clip_by_global_norm(tape.gradient(cost, train_weights), max_grad_norm)
            optimizer.apply_gradients(zip(grad, train_weights))
            costs += cost
            iters += 1
            if step % (epoch_size // 10) == 10:
                print('%.3f perplexity: %.3f speed: %.0f wps' % (step * 1.0 / epoch_size, np.exp(costs / iters), iters * batch_size * num_steps / (time.time() - start_time)))
        train_perplexity = np.exp(costs / iters)
        print('Epoch: %d/%d Train Perplexity: %.3f' % (i + 1, max_max_epoch, train_perplexity))
        net.eval()
        start_time = time.time()
        costs = 0.0
        iters = 0
        lstm1_state = None
        lstm2_state = None
        for (step, (x, y)) in enumerate(tl.iterate.ptb_iterator(valid_data, batch_size, num_steps)):
            (logits, lstm1_state, lstm2_state) = net(x, lstm1_initial_state=lstm1_state, lstm2_initial_state=lstm2_state)
            cost = tl.cost.cross_entropy(logits, tf.reshape(y, [-1]), name='train_loss')
            costs += cost
            iters += 1
        valid_perplexity = np.exp(costs / iters)
        print('Epoch: %d/%d Valid Perplexity: %.3f' % (i + 1, max_max_epoch, valid_perplexity))
    print('Evaluation')
    net.eval()
    start_time = time.time()
    costs = 0.0
    iters = 0
    lstm1_state = None
    lstm2_state = None
    for (step, (x, y)) in enumerate(tl.iterate.ptb_iterator(test_data, batch_size=1, num_steps=1)):
        (logits, lstm1_state, lstm2_state) = net(x, lstm1_initial_state=lstm1_state, lstm2_initial_state=lstm2_state)
        cost = tl.cost.cross_entropy(logits, tf.reshape(y, [-1]), name='train_loss')
        costs += cost
        iters += 1
    test_perplexity = np.exp(costs / iters)
    print('Test Perplexity: %.3f took %.2fs' % (test_perplexity, time.time() - start_time))
    print("More example: Text generation using Trump's speech data: https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_generate_text.py -- def main_lstm_generate_text():")
if __name__ == '__main__':
    main()