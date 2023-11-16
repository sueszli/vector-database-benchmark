import numpy as np

class HiddenMarkovModel(object):
    """
    Base class of Hidden Markov models
    """

    def __init__(self, initial_proba, transition_proba):
        if False:
            while True:
                i = 10
        '\n        construct hidden markov model\n\n        Parameters\n        ----------\n        initial_proba : (n_hidden,) np.ndarray\n            initial probability of each hidden state\n        transition_proba : (n_hidden, n_hidden) np.ndarray\n            transition probability matrix\n            (i, j) component denotes the transition probability from i-th to j-th hidden state\n\n        Attribute\n        ---------\n        n_hidden : int\n            number of hidden state\n        '
        self.n_hidden = initial_proba.size
        self.initial_proba = initial_proba
        self.transition_proba = transition_proba

    def fit(self, seq, iter_max=100):
        if False:
            i = 10
            return i + 15
        '\n        perform EM algorithm to estimate parameter of emission model and hidden variables\n\n        Parameters\n        ----------\n        seq : (N, ndim) np.ndarray\n            observed sequence\n        iter_max : int\n            maximum number of EM steps\n\n        Returns\n        -------\n        posterior : (N, n_hidden) np.ndarray\n            posterior distribution of each latent variable\n        '
        params = np.hstack((self.initial_proba.ravel(), self.transition_proba.ravel()))
        for i in range(iter_max):
            (p_hidden, p_transition) = self.expect(seq)
            self.maximize(seq, p_hidden, p_transition)
            params_new = np.hstack((self.initial_proba.ravel(), self.transition_proba.ravel()))
            if np.allclose(params, params_new):
                break
            else:
                params = params_new
        return self.forward_backward(seq)

    def expect(self, seq):
        if False:
            print('Hello World!')
        '\n        estimate posterior distributions of hidden states and\n        transition probability between adjacent latent variables\n\n        Parameters\n        ----------\n        seq : (N, ndim) np.ndarray\n            observed sequence\n\n        Returns\n        -------\n        p_hidden : (N, n_hidden) np.ndarray\n            posterior distribution of each hidden variable\n        p_transition : (N - 1, n_hidden, n_hidden) np.ndarray\n            posterior transition probability between adjacent latent variables\n        '
        likelihood = self.likelihood(seq)
        f = self.initial_proba * likelihood[0]
        constant = [f.sum()]
        forward = [f / f.sum()]
        for like in likelihood[1:]:
            f = forward[-1] @ self.transition_proba * like
            constant.append(f.sum())
            forward.append(f / f.sum())
        forward = np.asarray(forward)
        constant = np.asarray(constant)
        backward = [np.ones(self.n_hidden)]
        for (like, c) in zip(likelihood[-1:0:-1], constant[-1:0:-1]):
            backward.insert(0, self.transition_proba @ (like * backward[0]) / c)
        backward = np.asarray(backward)
        p_hidden = forward * backward
        p_transition = self.transition_proba * likelihood[1:, None, :] * backward[1:, None, :] * forward[:-1, :, None]
        return (p_hidden, p_transition)

    def forward_backward(self, seq):
        if False:
            while True:
                i = 10
        '\n        estimate posterior distributions of hidden states\n\n        Parameters\n        ----------\n        seq : (N, ndim) np.ndarray\n            observed sequence\n\n        Returns\n        -------\n        posterior : (N, n_hidden) np.ndarray\n            posterior distribution of hidden states\n        '
        likelihood = self.likelihood(seq)
        f = self.initial_proba * likelihood[0]
        constant = [f.sum()]
        forward = [f / f.sum()]
        for like in likelihood[1:]:
            f = forward[-1] @ self.transition_proba * like
            constant.append(f.sum())
            forward.append(f / f.sum())
        backward = [np.ones(self.n_hidden)]
        for (like, c) in zip(likelihood[-1:0:-1], constant[-1:0:-1]):
            backward.insert(0, self.transition_proba @ (like * backward[0]) / c)
        forward = np.asarray(forward)
        backward = np.asarray(backward)
        posterior = forward * backward
        return posterior

    def filtering(self, seq):
        if False:
            return 10
        '\n        bayesian filtering\n\n        Parameters\n        ----------\n        seq : (N, ndim) np.ndarray\n            observed sequence\n\n        Returns\n        -------\n        posterior : (N, n_hidden) np.ndarray\n            posterior distributions of each latent variables\n        '
        likelihood = self.likelihood(seq)
        p = self.initial_proba * likelihood[0]
        posterior = [p / np.sum(p)]
        for like in likelihood[1:]:
            p = posterior[-1] @ self.transition_proba * like
            posterior.append(p / np.sum(p))
        posterior = np.asarray(posterior)
        return posterior

    def viterbi(self, seq):
        if False:
            while True:
                i = 10
        '\n        viterbi algorithm (a.k.a. max-sum algorithm)\n\n        Parameters\n        ----------\n        seq : (N, ndim) np.ndarray\n            observed sequence\n\n        Returns\n        -------\n        seq_hid : (N,) np.ndarray\n            the most probable sequence of hidden variables\n        '
        nll = -np.log(self.likelihood(seq))
        cost_total = nll[0]
        from_list = []
        for i in range(1, len(seq)):
            cost_temp = cost_total[:, None] - np.log(self.transition_proba) + nll[i]
            cost_total = np.min(cost_temp, axis=0)
            index = np.argmin(cost_temp, axis=0)
            from_list.append(index)
        seq_hid = [np.argmin(cost_total)]
        for source in from_list[::-1]:
            seq_hid.insert(0, source[seq_hid[0]])
        return seq_hid