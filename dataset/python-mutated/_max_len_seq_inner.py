import numpy as np

def _max_len_seq_inner(taps, state, nbits, length, seq):
    if False:
        while True:
            i = 10
    n_taps = taps.shape[0]
    idx = 0
    for i in range(length):
        feedback = state[idx]
        seq[i] = feedback
        for ti in range(n_taps):
            feedback ^= state[(taps[ti] + idx) % nbits]
        state[idx] = feedback
        idx = (idx + 1) % nbits
    return np.roll(state, -idx, axis=0)