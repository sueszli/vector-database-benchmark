from .validation import check_random_state

def _init_arpack_v0(size, random_state):
    if False:
        while True:
            i = 10
    'Initialize the starting vector for iteration in ARPACK functions.\n\n    Initialize a ndarray with values sampled from the uniform distribution on\n    [-1, 1]. This initialization model has been chosen to be consistent with\n    the ARPACK one as another initialization can lead to convergence issues.\n\n    Parameters\n    ----------\n    size : int\n        The size of the eigenvalue vector to be initialized.\n\n    random_state : int, RandomState instance or None, default=None\n        The seed of the pseudo random number generator used to generate a\n        uniform distribution. If int, random_state is the seed used by the\n        random number generator; If RandomState instance, random_state is the\n        random number generator; If None, the random number generator is the\n        RandomState instance used by `np.random`.\n\n    Returns\n    -------\n    v0 : ndarray of shape (size,)\n        The initialized vector.\n    '
    random_state = check_random_state(random_state)
    v0 = random_state.uniform(-1, 1, size)
    return v0