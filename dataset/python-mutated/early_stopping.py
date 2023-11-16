"""
Example that trains an MLP using early stopping.
Training will stop when the stopping condition is satisfied
or when num_epochs has been reached, whichever is first.

Usage:

    python examples/early_stopping.py

"""
import os
from neon.data import MNIST
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
parser = NeonArgparser(__doc__)
args = parser.parse_args()
dataset = MNIST(path=args.data_dir)
train_set = dataset.train_iter
valid_set = dataset.valid_iter
init_norm = Gaussian(loc=0.0, scale=0.01)
layers = []
layers.append(Affine(nout=100, init=init_norm, batch_norm=True, activation=Rectlin()))
layers.append(Affine(nout=10, init=init_norm, activation=Logistic(shortcut=True)))
cost = GeneralizedCost(costfunc=CrossEntropyBinary())
mlp = Model(layers=layers)

def stop_func(s, v):
    if False:
        while True:
            i = 10
    if s is None:
        return (v, False)
    return (min(v, s), v > s)
optimizer = GradientDescentMomentum(learning_rate=0.1, momentum_coef=0.9)
if args.callback_args['eval_freq'] is None:
    args.callback_args['eval_freq'] = 1
callbacks = Callbacks(mlp, eval_set=valid_set, **args.callback_args)
callbacks.add_early_stop_callback(stop_func)
callbacks.add_save_best_state_callback(os.path.join(args.data_dir, 'early_stop-best_state.pkl'))
mlp.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)