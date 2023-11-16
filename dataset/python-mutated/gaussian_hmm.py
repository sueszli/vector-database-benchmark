import numpy as np
from prml.rv import MultivariateGaussian
from .hmm import HiddenMarkovModel

class GaussianHMM(HiddenMarkovModel):
    """
    Hidden Markov Model with Gaussian emission model
    """

    def __init__(self, initial_proba, transition_proba, means, covs):
        if False:
            for i in range(10):
                print('nop')
        '\n        construct hidden markov model with Gaussian emission model\n\n        Parameters\n        ----------\n        initial_proba : (n_hidden,) np.ndarray or None\n            probability of initial states\n        transition_proba : (n_hidden, n_hidden) np.ndarray or None\n            transition probability matrix\n            (i, j) component denotes the transition probability from i-th to j-th hidden state\n        means : (n_hidden, ndim) np.ndarray\n            mean of each gaussian component\n        covs : (n_hidden, ndim, ndim) np.ndarray\n            covariance matrix of each gaussian component\n\n        Attributes\n        ----------\n        ndim : int\n            dimensionality of observation space\n        n_hidden : int\n            number of hidden states\n        '
        assert initial_proba.size == transition_proba.shape[0] == transition_proba.shape[1] == means.shape[0] == covs.shape[0]
        assert means.shape[1] == covs.shape[1] == covs.shape[2]
        super().__init__(initial_proba, transition_proba)
        self.ndim = means.shape[1]
        self.means = means
        self.covs = covs
        self.precisions = np.linalg.inv(self.covs)
        self.gaussians = [MultivariateGaussian(m, cov) for (m, cov) in zip(means, covs)]

    def draw(self, n=100):
        if False:
            print('Hello World!')
        '\n        draw random sequence from this model\n\n        Parameters\n        ----------\n        n : int\n            length of the random sequence\n\n        Returns\n        -------\n        seq : (n, ndim) np.ndarray\n            generated random sequence\n        '
        hidden_state = np.random.choice(self.n_hidden, p=self.initial_proba)
        seq = []
        while len(seq) < n:
            seq.extend(self.gaussians[hidden_state].draw())
            hidden_state = np.random.choice(self.n_hidden, p=self.transition_proba[hidden_state])
        return np.asarray(seq)

    def likelihood(self, X):
        if False:
            while True:
                i = 10
        diff = X[:, None, :] - self.means
        exponents = np.sum(np.einsum('nki,kij->nkj', diff, self.precisions) * diff, axis=-1)
        return np.exp(-0.5 * exponents) / np.sqrt(np.linalg.det(self.covs) * (2 * np.pi) ** self.ndim)

    def maximize(self, seq, p_hidden, p_transition):
        if False:
            return 10
        self.initial_proba = p_hidden[0] / np.sum(p_hidden[0])
        self.transition_proba = np.sum(p_transition, axis=0) / np.sum(p_transition, axis=(0, 2))
        Nk = np.sum(p_hidden, axis=0)
        self.means = (seq.T @ p_hidden / Nk).T
        diffs = seq[:, None, :] - self.means
        self.covs = np.einsum('nki,nkj->kij', diffs, diffs * p_hidden[:, :, None]) / Nk[:, None, None]