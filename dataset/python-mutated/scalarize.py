"""
Copyright 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import List, Union
import cvxpy.atoms as atoms
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.transforms import indicator

def weighted_sum(objectives: List[Union[Minimize, Maximize]], weights) -> Union[Minimize, Maximize]:
    if False:
        while True:
            i = 10
    'Combines objectives as a weighted sum.\n\n    Args:\n      objectives: A list of Minimize/Maximize objectives.\n      weights: A vector of weights.\n\n    Returns:\n      A Minimize/Maximize objective.\n    '
    num_objs = len(objectives)
    return sum((objectives[i] * weights[i] for i in range(num_objs)))

def targets_and_priorities(objectives: List[Union[Minimize, Maximize]], priorities, targets, limits=None, off_target: float=1e-05) -> Union[Minimize, Maximize]:
    if False:
        while True:
            i = 10
    '\n    Combines objectives with penalties within a range between target and limit.\n\n    For nonnegative priorities, each Minimize objective i has value\n\n        off_target*objectives[i] when objectives[i] < targets[i]\n\n        (priorities[i]-off_target)*objectives[i] when targets[i] <= objectives[i] <= limits[i]\n\n        +infinity when objectives[i] > limits[i]\n\n    and each Maximize objective i has value\n\n        off_target*objectives[i] when objectives[i] > targets[i]\n\n        (priorities[i]-off_target)*objectives[i] when targets[i] >= objectives[i] >= limits[i]\n\n        -infinity when objectives[i] < limits[i]\n\n    A negative priority flips the objective sense, i.e., we \n    use -objectives[i], -targets[i], and -limits[i] with abs(priorities[i]).\n\n    Args:\n      objectives: A list of Minimize/Maximize objectives.\n      priorities: The weight within the trange.\n      targets: The start (end) of penalty for Minimize (Maximize)\n      limits: Optional hard end (start) of penalty for Minimize (Maximize)\n      off_target: Penalty outside of target.\n\n    Returns:\n      A Minimize/Maximize objective.\n\n    Raises:\n      ValueError: If the scalarized objective is neither convex nor concave.\n    '
    assert len(objectives) == len(priorities), 'Number of objectives and priorities must match.'
    assert len(objectives) == len(targets), 'Number of objectives and targets must match.'
    if limits is not None:
        assert len(objectives) == len(limits), 'Number of objectives and limits must match.'
    assert off_target >= 0, 'The off_target argument must be nonnegative.'
    num_objs = len(objectives)
    new_objs: List[Union[Minimize, Maximize]] = []
    for i in range(num_objs):
        (obj, tar, lim) = (objectives[i], targets[i], limits[i] if limits is not None else None)
        if priorities[i] < 0:
            (obj, tar, lim) = (-obj, -tar, -lim if lim is not None else None)
        sign = 1 if obj.args[0].is_convex() else -1
        delta = sign * (obj.args[0] - targets[i])
        expr = sign * (abs(priorities[i]) - off_target) * atoms.pos(delta)
        expr += off_target * obj.args[0]
        if limits is not None:
            expr += sign * indicator([sign * obj.args[0] <= sign * limits[i]])
        new_objs.append(expr)
    obj_expr = sum(new_objs)
    if obj_expr.is_convex():
        return Minimize(obj_expr)
    elif obj_expr.is_concave():
        return Maximize(obj_expr)
    else:
        raise ValueError('Scalarized objective is neither convex nor concave.')

def max(objectives: List[Union[Minimize, Maximize]], weights) -> Minimize:
    if False:
        print('Hello World!')
    'Combines objectives as max of weighted terms.\n\n    Args:\n      objectives: A list of Minimize/Maximize objectives.\n      weights: A vector of weights.\n\n    Returns:\n      A Minimize objective.\n    '
    num_objs = len(objectives)
    expr = atoms.maximum(*[(objectives[i] * weights[i]).args[0] for i in range(num_objs)])
    return Minimize(expr)

def log_sum_exp(objectives: List[Union[Minimize, Maximize]], weights, gamma: float=1.0) -> Minimize:
    if False:
        i = 10
        return i + 15
    'Combines objectives as log_sum_exp of weighted terms.\n\n\n    The objective takes the form\n        log(sum_{i=1}^n exp(gamma*weights[i]*objectives[i]))/gamma\n    As gamma goes to 0, log_sum_exp approaches weighted_sum. As gamma goes to infinity,\n    log_sum_exp approaches max.\n\n    Args:\n      objectives: A list of Minimize/Maximize objectives.\n      weights: A vector of weights.\n      gamma: Parameter interpolating between weighted_sum and max.\n\n    Returns:\n      A Minimize objective.\n    '
    num_objs = len(objectives)
    terms = [(objectives[i] * weights[i]).args[0] for i in range(num_objs)]
    expr = atoms.log_sum_exp(gamma * atoms.vstack(terms)) / gamma
    return Minimize(expr)