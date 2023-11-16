from typing import List
import numpy as np

def head_to_head_votes(ranks: List[List[int]]):
    if False:
        print('Hello World!')
    tallies = np.zeros((len(ranks[0]), len(ranks[0])))
    names = sorted(ranks[0])
    ranks = np.array(ranks)
    ranks = np.argsort(ranks, axis=1)
    for i in range(ranks.shape[1]):
        for j in range(i + 1, ranks.shape[1]):
            over_j = np.sum(ranks[:, i] < ranks[:, j])
            over_i = np.sum(ranks[:, j] < ranks[:, i])
            tallies[i, j] = over_j
            tallies[j, i] = over_i
    return (tallies, names)

def cycle_detect(pairs):
    if False:
        while True:
            i = 10
    'Recursively detect cycles by removing condorcet losers until either only one pair is left or condorcet losers no longer exist\n    This method upholds the invariant that in a ranking for all a,b either a>b or b>a for all a,b.\n\n\n    Returns\n    -------\n    out : False if the pairs do not contain a cycle, True if the pairs contain a cycle\n\n\n    '
    if len(pairs) <= 1:
        return False
    losers = [c_lose for c_lose in np.unique(pairs[:, 1]) if c_lose not in pairs[:, 0]]
    if len(losers) == 0:
        return True
    new = []
    for p in pairs:
        if p[1] not in losers:
            new.append(p)
    return cycle_detect(np.array(new))

def get_winner(pairs):
    if False:
        return 10
    '\n    This returns _one_ concordant winner.\n    It could be that there are multiple concordant winners, but in our case\n    since we are interested in a ranking, we have to choose one at random.\n    '
    losers = np.unique(pairs[:, 1]).astype(int)
    winners = np.unique(pairs[:, 0]).astype(int)
    for w in winners:
        if w not in losers:
            return w

def get_ranking(pairs):
    if False:
        print('Hello World!')
    '\n    Abuses concordance property to get a (not necessarily unique) ranking.\n    The lack of uniqueness is due to the potential existence of multiple\n    equally ranked winners. We have to pick one, which is where\n    the non-uniqueness comes from\n    '
    if len(pairs) == 1:
        return list(pairs[0])
    w = get_winner(pairs)
    p_new = np.array([(a, b) for (a, b) in pairs if a != w])
    return [w] + get_ranking(p_new)

def ranked_pairs(ranks: List[List[int]]):
    if False:
        for i in range(10):
            print('nop')
    '\n    Expects a list of rankings for an item like:\n        [("w","x","z","y") for _ in range(3)]\n        + [("w","y","x","z") for _ in range(2)]\n        + [("x","y","z","w") for _ in range(4)]\n        + [("x","z","w","y") for _ in range(5)]\n        + [("y","w","x","z") for _ in range(1)]\n    This code is quite brain melting, but the idea is the following:\n    1. create a head-to-head matrix that tallies up all win-lose combinations of preferences\n    2. take all combinations that win more than they loose and sort those by how often they win\n    3. use that to create an (implicit) directed graph\n    4. recursively extract nodes from the graph that do not have incoming edges\n    5. said recursive list is the ranking\n    '
    (tallies, names) = head_to_head_votes(ranks)
    tallies = tallies - tallies.T
    sorted_majorities = []
    for i in range(len(ranks[0])):
        for j in range(len(ranks[0])):
            if tallies[i, j] >= 0 and i != j:
                sorted_majorities.append((i, j, tallies[i, j]))
    sorted_majorities = np.array(sorted(sorted_majorities, key=lambda x: x[2], reverse=True))
    lock_ins = []
    for (x, y, _) in sorted_majorities:
        lock_ins.append((x, y))
        if cycle_detect(np.array(lock_ins)):
            lock_ins = lock_ins[:-1]
    numerical_ranks = np.array(get_ranking(np.array(lock_ins))).astype(int)
    conversion = [names[n] for n in numerical_ranks]
    return conversion
if __name__ == '__main__':
    ranks = ' (\n        [("w", "x", "z", "y") for _ in range(1)]\n        + [("w", "y", "x", "z") for _ in range(2)]\n        # + [("x","y","z","w") for _ in range(4)]\n        + [("x", "z", "w", "y") for _ in range(5)]\n        + [("y", "w", "x", "z") for _ in range(1)]\n        # [("y","z","w","x") for _ in range(1000)]\n    )'
    ranks = [['c5181083-d3e9-41e7-a935-83fb9fa01488', 'dcf3d179-0f34-4c15-ae21-b8feb15e422d', 'd11705af-5575-43e5-b22e-08d155fbaa62'], ['d11705af-5575-43e5-b22e-08d155fbaa62', 'c5181083-d3e9-41e7-a935-83fb9fa01488', 'dcf3d179-0f34-4c15-ae21-b8feb15e422d'], ['dcf3d179-0f34-4c15-ae21-b8feb15e422d', 'c5181083-d3e9-41e7-a935-83fb9fa01488', 'd11705af-5575-43e5-b22e-08d155fbaa62'], ['d11705af-5575-43e5-b22e-08d155fbaa62', 'c5181083-d3e9-41e7-a935-83fb9fa01488', 'dcf3d179-0f34-4c15-ae21-b8feb15e422d']]
    rp = ranked_pairs(ranks)
    print(rp)