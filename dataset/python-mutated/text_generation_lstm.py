"""
Train a LSTM based recurrent network on the Shakespeare dataset and
produce samples from the trained network.


Reference:

    Generating sequences with recurrent neural networks `[Graves2014]`_
.. _[Graves2014]: http://arxiv.org/pdf/1308.0850.pdf

Usage:

    python examples/text_generation_lstm.py

"""
import numpy as np
from neon import logger as neon_logger
from neon.data import Shakespeare
from neon.initializers import Uniform
from neon.layers import GeneralizedCost, LSTM, Affine
from neon.models import Model
from neon.optimizers import RMSProp
from neon.transforms import Logistic, Tanh, Softmax, CrossEntropyMulti
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
default_overrides = dict(save_path='rnn_text_gen.pickle', serialize=1, batch_size=64)
parser = NeonArgparser(__doc__, default_overrides=default_overrides)
args = parser.parse_args()
time_steps = 64
hidden_size = 512
gradient_clip_value = 5
dataset = Shakespeare(time_steps, path=args.data_dir)
train_set = dataset.train_iter
valid_set = dataset.valid_iter
init = Uniform(low=-0.08, high=0.08)
layers = [LSTM(hidden_size, init, activation=Logistic(), gate_activation=Tanh()), Affine(len(train_set.vocab), init, bias=init, activation=Softmax())]
model = Model(layers=layers)
cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))
optimizer = RMSProp(gradient_clip_value=gradient_clip_value, stochastic_round=args.rounding)
callbacks = Callbacks(model, eval_set=valid_set, **args.callback_args)
model.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

def sample(prob):
    if False:
        print('Hello World!')
    '\n    Sample index from probability distribution\n    '
    prob = prob / (prob.sum() + 1e-06)
    return np.argmax(np.random.multinomial(1, prob, 1))
model.be.bsz = 1
time_steps = 1
num_predict = 1000
layers = [LSTM(hidden_size, init, activation=Logistic(), gate_activation=Tanh()), Affine(len(train_set.vocab), init, bias=init, activation=Softmax())]
model_new = Model(layers=layers)
model_new.load_params(args.save_path)
model_new.initialize(dataset=(train_set.shape[0], time_steps))
text = []
seed_tokens = list('ROMEO:')
x = model_new.be.zeros((len(train_set.vocab), time_steps))
for s in seed_tokens:
    x.fill(0)
    x[train_set.token_to_index[s], 0] = 1
    y = model_new.fprop(x)
for i in range(num_predict):
    pred = sample(y.get()[:, -1])
    text.append(train_set.index_to_token[int(pred)])
    x.fill(0)
    x[int(pred), 0] = 1
    y = model_new.fprop(x)
neon_logger.display(''.join(seed_tokens + text))