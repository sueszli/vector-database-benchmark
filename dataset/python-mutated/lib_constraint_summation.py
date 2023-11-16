"""
lib_constraint_summation.py
"""
import math
import random
from operator import itemgetter

def check_feasibility(x_bounds, lowerbound, upperbound):
    if False:
        i = 10
        return i + 15
    '\n    This can have false positives.\n    For examples, parameters can only be 0 or 5, and the summation constraint is between 6 and 7.\n    '
    x_bounds_lowerbound = sum([x_bound[0] for x_bound in x_bounds])
    x_bounds_upperbound = sum([x_bound[-1] for x_bound in x_bounds])
    return x_bounds_lowerbound <= lowerbound <= x_bounds_upperbound or x_bounds_lowerbound <= upperbound <= x_bounds_upperbound

def rand(x_bounds, x_types, lowerbound, upperbound, max_retries=100):
    if False:
        return 10
    '\n    Key idea is that we try to move towards upperbound, by randomly choose one\n    value for each parameter. However, for the last parameter,\n    we need to make sure that its value can help us get above lowerbound\n    '
    outputs = None
    if check_feasibility(x_bounds, lowerbound, upperbound) is True:
        x_idx_sorted = []
        for (i, _) in enumerate(x_bounds):
            if x_types[i] == 'discrete_int':
                x_idx_sorted.append([i, len(x_bounds[i])])
            elif x_types[i] == 'range_int' or x_types[i] == 'range_continuous':
                x_idx_sorted.append([i, math.floor(x_bounds[i][1] - x_bounds[i][0])])
        x_idx_sorted = sorted(x_idx_sorted, key=itemgetter(1))
        for _ in range(max_retries):
            budget_allocated = 0
            outputs = [None] * len(x_bounds)
            for (i, _) in enumerate(x_idx_sorted):
                x_idx = x_idx_sorted[i][0]
                budget_max = upperbound - budget_allocated
                if i < len(x_idx_sorted) - 1:
                    if x_bounds[x_idx][0] <= budget_max:
                        if x_types[x_idx] == 'discrete_int':
                            temp = []
                            for j in x_bounds[x_idx]:
                                if j <= budget_max:
                                    temp.append(j)
                            if temp:
                                outputs[x_idx] = temp[random.randint(0, len(temp) - 1)]
                        elif x_types[x_idx] == 'range_int' or x_types[x_idx] == 'range_continuous':
                            outputs[x_idx] = random.randint(x_bounds[x_idx][0], min(x_bounds[x_idx][-1], budget_max))
                else:
                    randint_lowerbound = lowerbound - budget_allocated
                    randint_lowerbound = 0 if randint_lowerbound < 0 else randint_lowerbound
                    if x_bounds[x_idx][0] <= budget_max and x_bounds[x_idx][-1] >= randint_lowerbound:
                        if x_types[x_idx] == 'discrete_int':
                            temp = []
                            for j in x_bounds[x_idx]:
                                if randint_lowerbound <= j <= budget_max:
                                    temp.append(j)
                            if temp:
                                outputs[x_idx] = temp[random.randint(0, len(temp) - 1)]
                        elif x_types[x_idx] == 'range_int' or x_types[x_idx] == 'range_continuous':
                            outputs[x_idx] = random.randint(randint_lowerbound, min(x_bounds[x_idx][1], budget_max))
                if outputs[x_idx] is None:
                    break
                budget_allocated += outputs[x_idx]
            if None not in outputs:
                break
    return outputs