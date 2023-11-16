from hypothesis.internal.conjecture.shrinking.common import find_integer
FIND_INTEGER_COSTS = {}

def find_integer_cost(n):
    if False:
        print('Hello World!')
    try:
        return FIND_INTEGER_COSTS[n]
    except KeyError:
        pass
    cost = [0]

    def test(i):
        if False:
            for i in range(10):
                print('nop')
        cost[0] += 1
        return i <= n
    find_integer(test)
    return FIND_INTEGER_COSTS.setdefault(n, cost[0])