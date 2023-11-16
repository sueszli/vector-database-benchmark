import numpy as np
from scipy.special import gammaln, psi
from scipy.stats import dirichlet
'\nLegacy numerical solution.\nShould not be used as it is probably broken\n\n\ndef make_range(*x):\n    """\n    constructs leftover values for the simplex given the first k entries\n    (0,x_k) = 1-(x_1+...+x_(k-1))\n    """\n    return (0, max(0, 1 - sum(x)))\n\n\ndef relative_entropy(p, q):\n    """\n    relative entropy of the two given dirichlet distributions\n    """\n\n    def tmp(*x):\n        """\n        First adds the last always forced entry to the input (the last x_last = 1-(x_1+...+x_(N)) )\n        Then computes the relative entropy of posterior and prior for that datapoint\n        """\n        x_new = np.append(x, 1 - sum(x))\n        return p(x_new) * log2(p(x_new) / q(x_new))\n\n    return tmp\n\n\ndef naive_monte_carlo_integral(fun, dim, samples=10_000_000):\n    s = np.random.rand(dim - 1, samples)\n    s = np.sort(np.concatenate((np.zeros((1, samples)), s, np.ones((1, samples)))), 0)\n    # print(s)\n    pos = np.diff(s, axis=0)\n    # print(pos)\n    res = fun(pos)\n    return np.mean(res)\n\ndef infogain(a_post, a_prior):\n    raise (\n        """For the love of good don\'t use this:\n    it\'s insanely poorly conditioned, the worst numerical code I have ever written\n    and it\'s slow as molasses. Use the analytic solution instead.\n\n    Maybe remove\n    """\n    )\n    args = len(a_prior)\n    p = dirichlet(a_post).pdf\n    q = dirichlet(a_prior).pdf\n    (info, _) = nquad(relative_entropy(p, q), [make_range for _ in range(args - 1)], opts={"epsabs": 1e-8})\n    # info = naive_monte_carlo_integral(relative_entropy(p,q), len(a_post))\n    return info\n'

def analytic_solution(a_post, a_prior):
    if False:
        for i in range(10):
            print('nop')
    '\n    Analytic solution to the KL-divergence between two dirichlet distributions.\n    Proof is in the Notion design doc.\n    '
    post_sum = np.sum(a_post)
    prior_sum = np.sum(a_prior)
    info = gammaln(post_sum) - gammaln(prior_sum) - np.sum(gammaln(a_post)) + np.sum(gammaln(a_prior)) - np.sum((a_post - a_prior) * (psi(a_post) - psi(post_sum)))
    return info

def uniform_expected_infogain(a_prior):
    if False:
        for i in range(10):
            print('nop')
    mean_weight = dirichlet.mean(a_prior)
    results = []
    for (i, w) in enumerate(mean_weight):
        a_post = a_prior.copy()
        a_post[i] = a_post[i] + 1
        results.append(w * analytic_solution(a_post, a_prior))
    return np.sum(results)
if __name__ == '__main__':
    a_prior = np.array([1, 1, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    a_post = np.array([1, 1, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print('algebraic', analytic_solution(a_post, a_prior))
    print('large infogain', uniform_expected_infogain(a_prior))
    print('post infogain', uniform_expected_infogain(a_post))