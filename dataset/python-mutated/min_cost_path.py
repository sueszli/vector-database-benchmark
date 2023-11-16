"""
author @goswami-rahul

To find minimum cost path
from station 0 to station N-1,
where cost of moving from ith station to jth station is given as:

Matrix of size (N x N)
where Matrix[i][j] denotes the cost of moving from
station i --> station j   for i < j

NOTE that values where Matrix[i][j] and i > j does not
mean anything, and hence represented by -1 or INF

For the input below (cost matrix),
Minimum cost is obtained as from  { 0 --> 1 --> 3}
                                  = cost[0][1] + cost[1][3] = 65
the Output will be:

The Minimum cost to reach station 4 is 65

Time Complexity: O(n^2)
Space Complexity: O(n)
"""
INF = float('inf')

def min_cost(cost):
    if False:
        for i in range(10):
            print('nop')
    'Find minimum cost.\n\n    Keyword arguments:\n    cost -- matrix containing costs\n    '
    length = len(cost)
    dist = [INF] * length
    dist[0] = 0
    for i in range(length):
        for j in range(i + 1, length):
            dist[j] = min(dist[j], dist[i] + cost[i][j])
    return dist[length - 1]
if __name__ == '__main__':
    costs = [[0, 15, 80, 90], [-1, 0, 40, 50], [-1, -1, 0, 70], [-1, -1, -1, 0]]
    TOTAL_LEN = len(costs)
    mcost = min_cost(costs)
    assert mcost == 65
    print(f'The minimum cost to reach station {TOTAL_LEN} is {mcost}')