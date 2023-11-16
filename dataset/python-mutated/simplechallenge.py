from random import sample
import time
from golem.core.keysauth import get_random, sha2
__author__ = 'Magda.Stasiewicz'
CHALLENGE_HISTORY_LIMIT = 100
MAX_RANDINT = 100000000000000000000000000

def create_challenge(history, prev):
    if False:
        print('Hello World!')
    '\n    Creates puzzle by combining most recent puzzles solved, most recent puzzle challenged and random number history -\n    list of pairs node_id and most recent challenge given by this node prev - most recent challenge propagated by node\n    currently creating puzzle\n    '
    concat = ''
    for h in history:
        concat = concat + ''.join(sample(str(h[0]), min(CHALLENGE_HISTORY_LIMIT, len(h[0])))) + ''.join(sample(str(h[1]), min(CHALLENGE_HISTORY_LIMIT, len(h[1]))))
    if prev:
        concat += ''.join(sample(str(prev), min(CHALLENGE_HISTORY_LIMIT, len(prev))))
    concat += str(get_random(0, MAX_RANDINT))
    return concat

def solve_challenge(challenge, difficulty):
    if False:
        while True:
            i = 10
    "\n    Solves the puzzle given in string challenge difficulty is required number of zeros in the beginning of binary\n    representation of solution's hash returns solution and computation time in seconds\n    "
    start = time.time()
    min_hash = pow(2, 256 - difficulty)
    solution = 0
    while sha2(challenge + str(solution)) > min_hash:
        solution += 1
    end = time.time()
    return (solution, end - start)

def accept_challenge(challenge, solution, difficulty):
    if False:
        i = 10
        return i + 15
    ' Returns true if solution is valid for given challenge and difficulty, false otherwise\n    :param challenge:\n    :param solution:\n    :param int difficulty: difficulty of a challenge\n    :return boolean: true if solution is valid, false otherwise\n    '
    if sha2(challenge + str(solution)) <= pow(2, 256 - difficulty):
        return True
    return False