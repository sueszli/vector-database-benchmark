from .tsp import tsp_data

def change_dist(dist: dict, i: int, j: int, new_cost: float) -> float:
    if False:
        i = 10
        return i + 15
    'Change the distance between two points.\n\n    Args:\n        dist (dict): distance matrix, where the key is a pair and value is\n            the cost (aka, distance).\n        i (int): the source node\n        j (int): the destination node\n        new_cost (float): the new cost for the distance\n\n    Returns:\n        float: the previous cost\n    '
    prev_cost = dist[i, j]
    dist[i, j] = new_cost
    return prev_cost

def compare_costs(prev_cost, new_cost) -> float:
    if False:
        while True:
            i = 10
    'Compare the previous cost and the new cost.\n\n    Args:\n        prev_cost (float): the previous cost\n        new_cost (float): the updated cost\n\n    Returns:\n        float: the ratio between these two costs\n    '
    return (new_cost - prev_cost) / prev_cost
dists = tsp_data(5, seed=1)