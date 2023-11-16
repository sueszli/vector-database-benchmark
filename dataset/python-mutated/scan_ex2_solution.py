import theano
import theano.tensor as T
import numpy as np
probabilities = T.vector()
nb_samples = T.iscalar()
rng = T.shared_randomstreams.RandomStreams(1234)

def sample_from_pvect(pvect):
    if False:
        while True:
            i = 10
    ' Provided utility function: given a symbolic vector of\n    probabilities (which MUST sum to 1), sample one element\n    and return its index.\n    '
    onehot_sample = rng.multinomial(n=1, pvals=pvect)
    sample = onehot_sample.argmax()
    return sample

def set_p_to_zero(pvect, i):
    if False:
        while True:
            i = 10
    " Provided utility function: given a symbolic vector of\n    probabilities and an index 'i', set the probability of the\n    i-th element to 0 and renormalize the probabilities so they\n    sum to 1.\n    "
    new_pvect = T.set_subtensor(pvect[i], 0.0)
    new_pvect = new_pvect / new_pvect.sum()
    return new_pvect

def step(p):
    if False:
        while True:
            i = 10
    sample = sample_from_pvect(p)
    new_p = set_p_to_zero(p, sample)
    return (new_p, sample)
(output, updates) = theano.scan(fn=step, outputs_info=[probabilities, None], n_steps=nb_samples)
(modified_probabilities, samples) = output
f = theano.function(inputs=[probabilities, nb_samples], outputs=[samples], updates=updates)
test_probs = np.asarray([0.6, 0.3, 0.1], dtype=theano.config.floatX)
for i in range(10):
    print(f(test_probs, 2))