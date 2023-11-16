from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
X = 2 * np.random.randn(300) + np.sin(np.linspace(0, 3 * np.pi, 300))
plt.plot(X)
plt.title('original')
plt.show()
decay = T.scalar('decay')
sequence = T.vector('sequence')

def recurrence(x, last, decay):
    if False:
        i = 10
        return i + 15
    return (1 - decay) * x + decay * last
(outputs, _) = theano.scan(fn=recurrence, sequences=sequence, n_steps=sequence.shape[0], outputs_info=[np.float64(0)], non_sequences=[decay])
lpf = theano.function(inputs=[sequence, decay], outputs=outputs)
Y = lpf(X, 0.99)
plt.plot(Y)
plt.title('filtered')
plt.show()