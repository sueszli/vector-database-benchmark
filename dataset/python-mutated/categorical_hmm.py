import numpy as np
from .hmm import HiddenMarkovModel

class CategoricalHMM(HiddenMarkovModel):
    """
    Hidden Markov Model with categorical emission model
    """

    def __init__(self, initial_proba, transition_proba, means):
        if False:
            return 10
        '\n        construct hidden markov model with categorical emission model\n\n        Parameters\n        ----------\n        initial_proba : (n_hidden,) np.ndarray\n            probability of initial latent state\n        transition_proba : (n_hidden, n_hidden) np.ndarray\n            transition probability matrix\n            (i, j) component denotes the transition probability from i-th to j-th hidden state\n        means : (n_hidden, ndim) np.ndarray\n            mean parameters of categorical distribution\n\n        Returns\n        -------\n        ndim : int\n            number of observation categories\n        n_hidden : int\n            number of hidden states\n        '
        assert initial_proba.size == transition_proba.shape[0] == transition_proba.shape[1] == means.shape[0]
        assert np.allclose(means.sum(axis=1), 1)
        super().__init__(initial_proba, transition_proba)
        self.ndim = means.shape[1]
        self.means = means

    def draw(self, n=100):
        if False:
            print('Hello World!')
        '\n        draw random sequence from this model\n\n        Parameters\n        ----------\n        n : int\n            length of the random sequence\n\n        Returns\n        -------\n        seq : (n,) np.ndarray\n            generated random sequence\n        '
        hidden_state = np.random.choice(self.n_hidden, p=self.initial_proba)
        seq = []
        while len(seq) < n:
            seq.append(np.random.choice(self.ndim, p=self.means[hidden_state]))
            hidden_state = np.random.choice(self.n_hidden, p=self.transition_proba[hidden_state])
        return np.asarray(seq)

    def likelihood(self, X):
        if False:
            while True:
                i = 10
        return self.means[X]

    def maximize(self, seq, p_hidden, p_transition):
        if False:
            return 10
        self.initial_proba = p_hidden[0] / np.sum(p_hidden[0])
        self.transition_proba = np.sum(p_transition, axis=0) / np.sum(p_transition, axis=(0, 2))
        x = p_hidden[:, None, :] * np.eye(self.ndim)[seq][:, :, None]
        self.means = np.sum(x, axis=0) / np.sum(p_hidden, axis=0)