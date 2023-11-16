"""A state-emitting MarkovModel.

Note terminology similar to Manning and Schutze is used.


Functions:
train_bw        Train a markov model using the Baum-Welch algorithm.
train_visible   Train a visible markov model using MLE.
find_states     Find the a state sequence that explains some observations.

load            Load a MarkovModel.
save            Save a MarkovModel.

Classes:
MarkovModel     Holds the description of a markov model
"""
try:
    import numpy as np
except ImportError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Please install NumPy if you want to use Bio.MarkovModel. See http://www.numpy.org/') from None
logaddexp = np.logaddexp

def itemindex(values):
    if False:
        return 10
    'Return a dictionary of values with their sequence offset as keys.'
    d = {}
    entries = enumerate(values[::-1])
    n = len(values) - 1
    for (index, key) in entries:
        d[key] = n - index
    return d
np.random.seed()
VERY_SMALL_NUMBER = 1e-300
LOG0 = np.log(VERY_SMALL_NUMBER)

class MarkovModel:
    """Create a state-emitting MarkovModel object."""

    def __init__(self, states, alphabet, p_initial=None, p_transition=None, p_emission=None):
        if False:
            while True:
                i = 10
        'Initialize the class.'
        self.states = states
        self.alphabet = alphabet
        self.p_initial = p_initial
        self.p_transition = p_transition
        self.p_emission = p_emission

    def __str__(self):
        if False:
            i = 10
            return i + 15
        'Create a string representation of the MarkovModel object.'
        from io import StringIO
        handle = StringIO()
        save(self, handle)
        handle.seek(0)
        return handle.read()

def _readline_and_check_start(handle, start):
    if False:
        return 10
    'Read the first line and evaluate that begisn with the correct start (PRIVATE).'
    line = handle.readline()
    if not line.startswith(start):
        raise ValueError(f'I expected {start!r} but got {line!r}')
    return line

def load(handle):
    if False:
        while True:
            i = 10
    'Parse a file handle into a MarkovModel object.'
    line = _readline_and_check_start(handle, 'STATES:')
    states = line.split()[1:]
    line = _readline_and_check_start(handle, 'ALPHABET:')
    alphabet = line.split()[1:]
    mm = MarkovModel(states, alphabet)
    (N, M) = (len(states), len(alphabet))
    mm.p_initial = np.zeros(N)
    line = _readline_and_check_start(handle, 'INITIAL:')
    for i in range(len(states)):
        line = _readline_and_check_start(handle, f'  {states[i]}:')
        mm.p_initial[i] = float(line.split()[-1])
    mm.p_transition = np.zeros((N, N))
    line = _readline_and_check_start(handle, 'TRANSITION:')
    for i in range(len(states)):
        line = _readline_and_check_start(handle, f'  {states[i]}:')
        mm.p_transition[i, :] = [float(v) for v in line.split()[1:]]
    mm.p_emission = np.zeros((N, M))
    line = _readline_and_check_start(handle, 'EMISSION:')
    for i in range(len(states)):
        line = _readline_and_check_start(handle, f'  {states[i]}:')
        mm.p_emission[i, :] = [float(v) for v in line.split()[1:]]
    return mm

def save(mm, handle):
    if False:
        return 10
    'Save MarkovModel object into handle.'
    w = handle.write
    w(f"STATES: {' '.join(mm.states)}\n")
    w(f"ALPHABET: {' '.join(mm.alphabet)}\n")
    w('INITIAL:\n')
    for i in range(len(mm.p_initial)):
        w(f'  {mm.states[i]}: {mm.p_initial[i]:g}\n')
    w('TRANSITION:\n')
    for i in range(len(mm.p_transition)):
        w(f"  {mm.states[i]}: {' '.join((str(x) for x in mm.p_transition[i]))}\n")
    w('EMISSION:\n')
    for i in range(len(mm.p_emission)):
        w(f"  {mm.states[i]}: {' '.join((str(x) for x in mm.p_emission[i]))}\n")

def train_bw(states, alphabet, training_data, pseudo_initial=None, pseudo_transition=None, pseudo_emission=None, update_fn=None):
    if False:
        for i in range(10):
            print('nop')
    'Train a MarkovModel using the Baum-Welch algorithm.\n\n    Train a MarkovModel using the Baum-Welch algorithm.  states is a list\n    of strings that describe the names of each state.  alphabet is a\n    list of objects that indicate the allowed outputs.  training_data\n    is a list of observations.  Each observation is a list of objects\n    from the alphabet.\n\n    pseudo_initial, pseudo_transition, and pseudo_emission are\n    optional parameters that you can use to assign pseudo-counts to\n    different matrices.  They should be matrices of the appropriate\n    size that contain numbers to add to each parameter matrix, before\n    normalization.\n\n    update_fn is an optional callback that takes parameters\n    (iteration, log_likelihood).  It is called once per iteration.\n    '
    (N, M) = (len(states), len(alphabet))
    if not training_data:
        raise ValueError('No training data given.')
    if pseudo_initial is not None:
        pseudo_initial = np.asarray(pseudo_initial)
        if pseudo_initial.shape != (N,):
            raise ValueError('pseudo_initial not shape len(states)')
    if pseudo_transition is not None:
        pseudo_transition = np.asarray(pseudo_transition)
        if pseudo_transition.shape != (N, N):
            raise ValueError('pseudo_transition not shape len(states) X len(states)')
    if pseudo_emission is not None:
        pseudo_emission = np.asarray(pseudo_emission)
        if pseudo_emission.shape != (N, M):
            raise ValueError('pseudo_emission not shape len(states) X len(alphabet)')
    training_outputs = []
    indexes = itemindex(alphabet)
    for outputs in training_data:
        training_outputs.append([indexes[x] for x in outputs])
    lengths = [len(x) for x in training_outputs]
    if min(lengths) == 0:
        raise ValueError('I got training data with outputs of length 0')
    x = _baum_welch(N, M, training_outputs, pseudo_initial=pseudo_initial, pseudo_transition=pseudo_transition, pseudo_emission=pseudo_emission, update_fn=update_fn)
    (p_initial, p_transition, p_emission) = x
    return MarkovModel(states, alphabet, p_initial, p_transition, p_emission)
MAX_ITERATIONS = 1000

def _baum_welch(N, M, training_outputs, p_initial=None, p_transition=None, p_emission=None, pseudo_initial=None, pseudo_transition=None, pseudo_emission=None, update_fn=None):
    if False:
        print('Hello World!')
    'Implement the Baum-Welch algorithm to evaluate unknown parameters in the MarkovModel object (PRIVATE).'
    if p_initial is None:
        p_initial = _random_norm(N)
    else:
        p_initial = _copy_and_check(p_initial, (N,))
    if p_transition is None:
        p_transition = _random_norm((N, N))
    else:
        p_transition = _copy_and_check(p_transition, (N, N))
    if p_emission is None:
        p_emission = _random_norm((N, M))
    else:
        p_emission = _copy_and_check(p_emission, (N, M))
    lp_initial = np.log(p_initial)
    lp_transition = np.log(p_transition)
    lp_emission = np.log(p_emission)
    if pseudo_initial is not None:
        lpseudo_initial = np.log(pseudo_initial)
    else:
        lpseudo_initial = None
    if pseudo_transition is not None:
        lpseudo_transition = np.log(pseudo_transition)
    else:
        lpseudo_transition = None
    if pseudo_emission is not None:
        lpseudo_emission = np.log(pseudo_emission)
    else:
        lpseudo_emission = None
    prev_llik = None
    for i in range(MAX_ITERATIONS):
        llik = LOG0
        for outputs in training_outputs:
            llik += _baum_welch_one(N, M, outputs, lp_initial, lp_transition, lp_emission, lpseudo_initial, lpseudo_transition, lpseudo_emission)
        if update_fn is not None:
            update_fn(i, llik)
        if prev_llik is not None and np.fabs(prev_llik - llik) < 0.1:
            break
        prev_llik = llik
    else:
        raise RuntimeError('HMM did not converge in %d iterations' % MAX_ITERATIONS)
    return [np.exp(_) for _ in (lp_initial, lp_transition, lp_emission)]

def _baum_welch_one(N, M, outputs, lp_initial, lp_transition, lp_emission, lpseudo_initial, lpseudo_transition, lpseudo_emission):
    if False:
        while True:
            i = 10
    'Execute one step for Baum-Welch algorithm (PRIVATE).\n\n    Do one iteration of Baum-Welch based on a sequence of output.\n    Changes the value for lp_initial, lp_transition and lp_emission in place.\n    '
    T = len(outputs)
    fmat = _forward(N, T, lp_initial, lp_transition, lp_emission, outputs)
    bmat = _backward(N, T, lp_transition, lp_emission, outputs)
    lp_arc = np.zeros((N, N, T))
    for t in range(T):
        k = outputs[t]
        lp_traverse = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                lp = fmat[i][t] + lp_transition[i][j] + lp_emission[i][k] + bmat[j][t + 1]
                lp_traverse[i][j] = lp
        lp_arc[:, :, t] = lp_traverse - _logsum(lp_traverse)
    lp_arcout_t = np.zeros((N, T))
    for t in range(T):
        for i in range(N):
            lp_arcout_t[i][t] = _logsum(lp_arc[i, :, t])
    lp_arcout = np.zeros(N)
    for i in range(N):
        lp_arcout[i] = _logsum(lp_arcout_t[i, :])
    lp_initial = lp_arcout_t[:, 0]
    if lpseudo_initial is not None:
        lp_initial = _logvecadd(lp_initial, lpseudo_initial)
        lp_initial = lp_initial - _logsum(lp_initial)
    for i in range(N):
        for j in range(N):
            lp_transition[i][j] = _logsum(lp_arc[i, j, :]) - lp_arcout[i]
        if lpseudo_transition is not None:
            lp_transition[i] = _logvecadd(lp_transition[i], lpseudo_transition)
            lp_transition[i] = lp_transition[i] - _logsum(lp_transition[i])
    for i in range(N):
        ksum = np.zeros(M) + LOG0
        for t in range(T):
            k = outputs[t]
            for j in range(N):
                ksum[k] = logaddexp(ksum[k], lp_arc[i, j, t])
        ksum = ksum - _logsum(ksum)
        if lpseudo_emission is not None:
            ksum = _logvecadd(ksum, lpseudo_emission[i])
            ksum = ksum - _logsum(ksum)
        lp_emission[i, :] = ksum
    return _logsum(fmat[:, T])

def _forward(N, T, lp_initial, lp_transition, lp_emission, outputs):
    if False:
        for i in range(10):
            print('nop')
    'Implement forward algorithm (PRIVATE).\n\n    Calculate a Nx(T+1) matrix, where the last column is the total\n    probability of the output.\n    '
    matrix = np.zeros((N, T + 1))
    matrix[:, 0] = lp_initial
    for t in range(1, T + 1):
        k = outputs[t - 1]
        for j in range(N):
            lprob = LOG0
            for i in range(N):
                lp = matrix[i][t - 1] + lp_transition[i][j] + lp_emission[i][k]
                lprob = logaddexp(lprob, lp)
            matrix[j][t] = lprob
    return matrix

def _backward(N, T, lp_transition, lp_emission, outputs):
    if False:
        print('Hello World!')
    'Implement backward algorithm (PRIVATE).'
    matrix = np.zeros((N, T + 1))
    for t in range(T - 1, -1, -1):
        k = outputs[t]
        for i in range(N):
            lprob = LOG0
            for j in range(N):
                lp = matrix[j][t + 1] + lp_transition[i][j] + lp_emission[i][k]
                lprob = logaddexp(lprob, lp)
            matrix[i][t] = lprob
    return matrix

def train_visible(states, alphabet, training_data, pseudo_initial=None, pseudo_transition=None, pseudo_emission=None):
    if False:
        print('Hello World!')
    'Train a visible MarkovModel using maximum likelihoood estimates for each of the parameters.\n\n    Train a visible MarkovModel using maximum likelihoood estimates\n    for each of the parameters.  states is a list of strings that\n    describe the names of each state.  alphabet is a list of objects\n    that indicate the allowed outputs.  training_data is a list of\n    (outputs, observed states) where outputs is a list of the emission\n    from the alphabet, and observed states is a list of states from\n    states.\n\n    pseudo_initial, pseudo_transition, and pseudo_emission are\n    optional parameters that you can use to assign pseudo-counts to\n    different matrices.  They should be matrices of the appropriate\n    size that contain numbers to add to each parameter matrix.\n    '
    (N, M) = (len(states), len(alphabet))
    if pseudo_initial is not None:
        pseudo_initial = np.asarray(pseudo_initial)
        if pseudo_initial.shape != (N,):
            raise ValueError('pseudo_initial not shape len(states)')
    if pseudo_transition is not None:
        pseudo_transition = np.asarray(pseudo_transition)
        if pseudo_transition.shape != (N, N):
            raise ValueError('pseudo_transition not shape len(states) X len(states)')
    if pseudo_emission is not None:
        pseudo_emission = np.asarray(pseudo_emission)
        if pseudo_emission.shape != (N, M):
            raise ValueError('pseudo_emission not shape len(states) X len(alphabet)')
    (training_states, training_outputs) = ([], [])
    states_indexes = itemindex(states)
    outputs_indexes = itemindex(alphabet)
    for (toutputs, tstates) in training_data:
        if len(tstates) != len(toutputs):
            raise ValueError('states and outputs not aligned')
        training_states.append([states_indexes[x] for x in tstates])
        training_outputs.append([outputs_indexes[x] for x in toutputs])
    x = _mle(N, M, training_outputs, training_states, pseudo_initial, pseudo_transition, pseudo_emission)
    (p_initial, p_transition, p_emission) = x
    return MarkovModel(states, alphabet, p_initial, p_transition, p_emission)

def _mle(N, M, training_outputs, training_states, pseudo_initial, pseudo_transition, pseudo_emission):
    if False:
        while True:
            i = 10
    'Implement Maximum likelihood estimation algorithm (PRIVATE).'
    p_initial = np.zeros(N)
    if pseudo_initial:
        p_initial = p_initial + pseudo_initial
    for states in training_states:
        p_initial[states[0]] += 1
    p_initial = _normalize(p_initial)
    p_transition = np.zeros((N, N))
    if pseudo_transition:
        p_transition = p_transition + pseudo_transition
    for states in training_states:
        for n in range(len(states) - 1):
            (i, j) = (states[n], states[n + 1])
            p_transition[i, j] += 1
    for i in range(len(p_transition)):
        p_transition[i, :] = p_transition[i, :] / sum(p_transition[i, :])
    p_emission = np.zeros((N, M))
    if pseudo_emission:
        p_emission = p_emission + pseudo_emission
    p_emission = np.ones((N, M))
    for (outputs, states) in zip(training_outputs, training_states):
        for (o, s) in zip(outputs, states):
            p_emission[s, o] += 1
    for i in range(len(p_emission)):
        p_emission[i, :] = p_emission[i, :] / sum(p_emission[i, :])
    return (p_initial, p_transition, p_emission)

def _argmaxes(vector, allowance=None):
    if False:
        i = 10
        return i + 15
    'Return indices of the maximum values aong the vector (PRIVATE).'
    return [np.argmax(vector)]

def find_states(markov_model, output):
    if False:
        i = 10
        return i + 15
    'Find states in the given Markov model output.\n\n    Returns a list of (states, score) tuples.\n    '
    mm = markov_model
    N = len(mm.states)
    lp_initial = np.log(mm.p_initial + VERY_SMALL_NUMBER)
    lp_transition = np.log(mm.p_transition + VERY_SMALL_NUMBER)
    lp_emission = np.log(mm.p_emission + VERY_SMALL_NUMBER)
    indexes = itemindex(mm.alphabet)
    output = [indexes[x] for x in output]
    results = _viterbi(N, lp_initial, lp_transition, lp_emission, output)
    for i in range(len(results)):
        (states, score) = results[i]
        results[i] = ([mm.states[x] for x in states], np.exp(score))
    return results

def _viterbi(N, lp_initial, lp_transition, lp_emission, output):
    if False:
        return 10
    'Implement Viterbi algorithm to find most likely states for a given input (PRIVATE).'
    T = len(output)
    backtrace = []
    for i in range(N):
        backtrace.append([None] * T)
    scores = np.zeros((N, T))
    scores[:, 0] = lp_initial + lp_emission[:, output[0]]
    for t in range(1, T):
        k = output[t]
        for j in range(N):
            i_scores = scores[:, t - 1] + lp_transition[:, j] + lp_emission[j, k]
            indexes = _argmaxes(i_scores)
            scores[j, t] = i_scores[indexes[0]]
            backtrace[j][t] = indexes
    in_process = []
    results = []
    indexes = _argmaxes(scores[:, T - 1])
    for i in indexes:
        in_process.append((T - 1, [i], scores[i][T - 1]))
    while in_process:
        (t, states, score) = in_process.pop()
        if t == 0:
            results.append((states, score))
        else:
            indexes = backtrace[states[0]][t]
            for i in indexes:
                in_process.append((t - 1, [i] + states, score))
    return results

def _normalize(matrix):
    if False:
        i = 10
        return i + 15
    'Normalize matrix object (PRIVATE).'
    if len(matrix.shape) == 1:
        matrix = matrix / sum(matrix)
    elif len(matrix.shape) == 2:
        for i in range(len(matrix)):
            matrix[i, :] = matrix[i, :] / sum(matrix[i, :])
    else:
        raise ValueError('I cannot handle matrixes of that shape')
    return matrix

def _uniform_norm(shape):
    if False:
        for i in range(10):
            print('nop')
    'Normalize a uniform matrix (PRIVATE).'
    matrix = np.ones(shape)
    return _normalize(matrix)

def _random_norm(shape):
    if False:
        return 10
    'Normalize a random matrix (PRIVATE).'
    matrix = np.random.random(shape)
    return _normalize(matrix)

def _copy_and_check(matrix, desired_shape):
    if False:
        while True:
            i = 10
    'Copy a matrix and check its dimension. Normalize at the end (PRIVATE).'
    matrix = np.array(matrix, copy=1)
    if matrix.shape != desired_shape:
        raise ValueError('Incorrect dimension')
    if len(matrix.shape) == 1:
        if np.fabs(sum(matrix) - 1.0) > 0.01:
            raise ValueError('matrix not normalized to 1.0')
    elif len(matrix.shape) == 2:
        for i in range(len(matrix)):
            if np.fabs(sum(matrix[i]) - 1.0) > 0.01:
                raise ValueError('matrix %d not normalized to 1.0' % i)
    else:
        raise ValueError("I don't handle matrices > 2 dimensions")
    return matrix

def _logsum(matrix):
    if False:
        return 10
    'Implement logsum for a matrix object (PRIVATE).'
    if len(matrix.shape) > 1:
        vec = np.reshape(matrix, (np.prod(matrix.shape),))
    else:
        vec = matrix
    sum = LOG0
    for num in vec:
        sum = logaddexp(sum, num)
    return sum

def _logvecadd(logvec1, logvec2):
    if False:
        return 10
    'Implement a log sum for two vector objects (PRIVATE).'
    assert len(logvec1) == len(logvec2), "vectors aren't the same length"
    sumvec = np.zeros(len(logvec1))
    for i in range(len(logvec1)):
        sumvec[i] = logaddexp(logvec1[i], logvec2[i])
    return sumvec

def _exp_logsum(numbers):
    if False:
        return 10
    'Return the exponential of a logsum (PRIVATE).'
    sum = _logsum(numbers)
    return np.exp(sum)