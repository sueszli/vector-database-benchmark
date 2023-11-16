from __future__ import print_function, division
from builtins import range
import numpy as np
import theano
import theano.tensor as T
N = T.iscalar('N')

def recurrence(n, fn_1, fn_2):
    if False:
        return 10
    return (fn_1 + fn_2, fn_1)
(outputs, updates) = theano.scan(fn=recurrence, sequences=T.arange(N), n_steps=N, outputs_info=[1.0, 1.0])
fibonacci = theano.function(inputs=[N], outputs=outputs)
o_val = fibonacci(8)
print('output:', o_val)