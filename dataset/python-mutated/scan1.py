from __future__ import print_function, division
from builtins import range
import numpy as np
import theano
import theano.tensor as T
x = T.vector('x')

def square(x):
    if False:
        while True:
            i = 10
    return x * x
(outputs, updates) = theano.scan(fn=square, sequences=x, n_steps=x.shape[0])
square_op = theano.function(inputs=[x], outputs=[outputs])
o_val = square_op(np.array([1, 2, 3, 4, 5]))
print('output:', o_val)