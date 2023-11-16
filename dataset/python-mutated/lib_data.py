import math
import random

def match_val_type(vals, vals_bounds, vals_types):
    if False:
        for i in range(10):
            print('nop')
    '\n    Update values in the array, to match their corresponding type\n    '
    vals_new = []
    for (i, _) in enumerate(vals_types):
        if vals_types[i] == 'discrete_int':
            vals_new.append(min(vals_bounds[i], key=lambda x: abs(x - vals[i])))
        elif vals_types[i] == 'range_int':
            vals_new.append(math.floor(vals[i]))
        elif vals_types[i] == 'range_continuous':
            vals_new.append(vals[i])
        else:
            return None
    return vals_new

def rand(x_bounds, x_types):
    if False:
        i = 10
        return i + 15
    '\n    Random generate variable value within their bounds\n    '
    outputs = []
    for (i, _) in enumerate(x_bounds):
        if x_types[i] == 'discrete_int':
            temp = x_bounds[i][random.randint(0, len(x_bounds[i]) - 1)]
            outputs.append(temp)
        elif x_types[i] == 'range_int':
            temp = random.randint(x_bounds[i][0], x_bounds[i][1] - 1)
            outputs.append(temp)
        elif x_types[i] == 'range_continuous':
            temp = random.uniform(x_bounds[i][0], x_bounds[i][1])
            outputs.append(temp)
        else:
            return None
    return outputs