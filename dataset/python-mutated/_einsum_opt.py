import itertools

def _flop_count(idx_contraction, inner, num_terms, size_dictionary):
    if False:
        for i in range(10):
            print('nop')
    "Copied from _flop_count in numpy/core/einsumfunc.py\n\n    Computes the number of FLOPS in the contraction.\n\n    Parameters\n    ----------\n    idx_contraction : iterable\n        The indices involved in the contraction\n    inner : bool\n        Does this contraction require an inner product?\n    num_terms : int\n        The number of terms in a contraction\n    size_dictionary : dict\n        The size of each of the indices in idx_contraction\n\n    Returns\n    -------\n    flop_count : int\n        The total number of FLOPS required for the contraction.\n\n    Examples\n    --------\n\n    >>> _flop_count('abc', False, 1, {'a': 2, 'b':3, 'c':5})\n    90\n\n    >>> _flop_count('abc', True, 2, {'a': 2, 'b':3, 'c':5})\n    270\n\n    "
    overall_size = _compute_size_by_dict(idx_contraction, size_dictionary)
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1
    return overall_size * op_factor

def _compute_size_by_dict(indices, idx_dict):
    if False:
        while True:
            i = 10
    "Copied from _compute_size_by_dict in numpy/core/einsumfunc.py\n\n    Computes the product of the elements in indices based on the dictionary\n    idx_dict.\n\n    Parameters\n    ----------\n    indices : iterable\n        Indices to base the product on.\n    idx_dict : dictionary\n        Dictionary of index sizes\n\n    Returns\n    -------\n    ret : int\n        The resulting product.\n\n    Examples\n    --------\n    >>> _compute_size_by_dict('abbc', {'a': 2, 'b':3, 'c':5})\n    90\n\n    "
    ret = 1
    for i in indices:
        ret *= idx_dict[i]
    return ret

def _find_contraction(positions, input_sets, output_set):
    if False:
        while True:
            i = 10
    "Copied from _find_contraction in numpy/core/einsumfunc.py\n\n    Finds the contraction for a given set of input and output sets.\n\n    Parameters\n    ----------\n    positions : iterable\n        Integer positions of terms used in the contraction.\n    input_sets : list\n        List of sets that represent the lhs side of the einsum subscript\n    output_set : set\n        Set that represents the rhs side of the overall einsum subscript\n\n    Returns\n    -------\n    new_result : set\n        The indices of the resulting contraction\n    remaining : list\n        List of sets that have not been contracted, the new set is appended to\n        the end of this list\n    idx_removed : set\n        Indices removed from the entire contraction\n    idx_contraction : set\n        The indices used in the current contraction\n\n    Examples\n    --------\n\n    # A simple dot product test case\n    >>> pos = (0, 1)\n    >>> isets = [set('ab'), set('bc')]\n    >>> oset = set('ac')\n    >>> _find_contraction(pos, isets, oset)\n    ({'a', 'c'}, [{'a', 'c'}], {'b'}, {'a', 'b', 'c'})\n\n    # A more complex case with additional terms in the contraction\n    >>> pos = (0, 2)\n    >>> isets = [set('abd'), set('ac'), set('bdc')]\n    >>> oset = set('ac')\n    >>> _find_contraction(pos, isets, oset)\n    ({'a', 'c'}, [{'a', 'c'}, {'a', 'c'}], {'b', 'd'}, {'a', 'b', 'c', 'd'})\n    "
    idx_contract = set()
    idx_remain = output_set.copy()
    remaining = []
    for (ind, value) in enumerate(input_sets):
        if ind in positions:
            idx_contract |= value
        else:
            remaining.append(value)
            idx_remain |= value
    new_result = idx_remain & idx_contract
    idx_removed = idx_contract - new_result
    remaining.append(new_result)
    return (new_result, remaining, idx_removed, idx_contract)

def _optimal_path(input_sets, output_set, idx_dict, memory_limit):
    if False:
        return 10
    "Copied from _optimal_path in numpy/core/einsumfunc.py\n\n    Computes all possible pair contractions, sieves the results based\n    on ``memory_limit`` and returns the lowest cost path. This algorithm\n    scales factorial with respect to the elements in the list ``input_sets``.\n\n    Parameters\n    ----------\n    input_sets : list\n        List of sets that represent the lhs side of the einsum subscript\n    output_set : set\n        Set that represents the rhs side of the overall einsum subscript\n    idx_dict : dictionary\n        Dictionary of index sizes\n    memory_limit : int\n        The maximum number of elements in a temporary array\n\n    Returns\n    -------\n    path : list\n        The optimal contraction order within the memory limit constraint.\n\n    Examples\n    --------\n    >>> isets = [set('abd'), set('ac'), set('bdc')]\n    >>> oset = set('')\n    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}\n    >>> _optimal_path(isets, oset, idx_sizes, 5000)\n    [(0, 2), (0, 1)]\n    "
    full_results = [(0, [], input_sets)]
    for iteration in range(len(input_sets) - 1):
        iter_results = []
        for curr in full_results:
            (cost, positions, remaining) = curr
            for con in itertools.combinations(range(len(input_sets) - iteration), 2):
                cont = _find_contraction(con, remaining, output_set)
                (new_result, new_input_sets, idx_removed, idx_contract) = cont
                new_size = _compute_size_by_dict(new_result, idx_dict)
                if new_size > memory_limit:
                    continue
                total_cost = cost + _flop_count(idx_contract, idx_removed, len(con), idx_dict)
                new_pos = positions + [con]
                iter_results.append((total_cost, new_pos, new_input_sets))
        if iter_results:
            full_results = iter_results
        else:
            path = min(full_results, key=lambda x: x[0])[1]
            path += [tuple(range(len(input_sets) - iteration))]
            return path
    if len(full_results) == 0:
        return [tuple(range(len(input_sets)))]
    path = min(full_results, key=lambda x: x[0])[1]
    return path

def _parse_possible_contraction(positions, input_sets, output_set, idx_dict, memory_limit, path_cost, naive_cost):
    if False:
        print('Hello World!')
    'Copied from _parse_possible_contraction in numpy/core/einsumfunc.py\n\n    Compute the cost (removed size + flops) and resultant indices for\n    performing the contraction specified by ``positions``.\n\n    Parameters\n    ----------\n    positions : tuple of int\n        The locations of the proposed tensors to contract.\n    input_sets : list of sets\n        The indices found on each tensors.\n    output_set : set\n        The output indices of the expression.\n    idx_dict : dict\n        Mapping of each index to its size.\n    memory_limit : int\n        The total allowed size for an intermediary tensor.\n    path_cost : int\n        The contraction cost so far.\n    naive_cost : int\n        The cost of the unoptimized expression.\n\n    Returns\n    -------\n    cost : (int, int)\n        A tuple containing the size of any indices removed, and the flop cost.\n    positions : tuple of int\n        The locations of the proposed tensors to contract.\n    new_input_sets : list of sets\n        The resulting new list of indices if this proposed contraction is performed.\n\n    '
    contract = _find_contraction(positions, input_sets, output_set)
    (idx_result, new_input_sets, idx_removed, idx_contract) = contract
    new_size = _compute_size_by_dict(idx_result, idx_dict)
    if new_size > memory_limit:
        return None
    old_sizes = (_compute_size_by_dict(input_sets[p], idx_dict) for p in positions)
    removed_size = sum(old_sizes) - new_size
    cost = _flop_count(idx_contract, idx_removed, len(positions), idx_dict)
    sort = (-removed_size, cost)
    if path_cost + cost > naive_cost:
        return None
    return [sort, positions, new_input_sets]

def _update_other_results(results, best):
    if False:
        while True:
            i = 10
    'Copied from _update_other_results in numpy/core/einsumfunc.py\n\n    Update the positions and provisional input_sets of ``results`` based on\n    performing the contraction result ``best``. Remove any involving the tensors\n    contracted.\n\n    Parameters\n    ----------\n    results : list\n        List of contraction results produced by ``_parse_possible_contraction``.\n    best : list\n        The best contraction of ``results`` i.e. the one that will be performed.\n\n    Returns\n    -------\n    mod_results : list\n        The list of modifed results, updated with outcome of ``best`` contraction.  # NOQA\n    '
    best_con = best[1]
    (bx, by) = best_con
    mod_results = []
    for (cost, (x, y), con_sets) in results:
        if x in best_con or y in best_con:
            continue
        del con_sets[by - int(by > x) - int(by > y)]
        del con_sets[bx - int(bx > x) - int(bx > y)]
        con_sets.insert(-1, best[2][-1])
        mod_con = (x - int(x > bx) - int(x > by), y - int(y > bx) - int(y > by))
        mod_results.append((cost, mod_con, con_sets))
    return mod_results

def _greedy_path(input_sets, output_set, idx_dict, memory_limit):
    if False:
        return 10
    "Copied from _greedy_path in numpy/core/einsumfunc.py\n\n    Finds the path by contracting the best pair until the input list is\n    exhausted. The best pair is found by minimizing the tuple\n    ``(-prod(indices_removed), cost)``.  What this amounts to is prioritizing\n    matrix multiplication or inner product operations, then Hadamard like\n    operations, and finally outer operations. Outer products are limited by\n    ``memory_limit``. This algorithm scales cubically with respect to the\n    number of elements in the list ``input_sets``.\n\n    Parameters\n    ----------\n    input_sets : list\n        List of sets that represent the lhs side of the einsum subscript\n    output_set : set\n        Set that represents the rhs side of the overall einsum subscript\n    idx_dict : dictionary\n        Dictionary of index sizes\n    memory_limit_limit : int\n        The maximum number of elements in a temporary array\n\n    Returns\n    -------\n    path : list\n        The greedy contraction order within the memory limit constraint.\n\n    Examples\n    --------\n    >>> isets = [set('abd'), set('ac'), set('bdc')]\n    >>> oset = set('')\n    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}\n    >>> _greedy_path(isets, oset, idx_sizes, 5000)\n    [(0, 2), (0, 1)]\n    "
    if len(input_sets) == 1:
        return [(0,)]
    elif len(input_sets) == 2:
        return [(0, 1)]
    contract = _find_contraction(range(len(input_sets)), input_sets, output_set)
    (idx_result, new_input_sets, idx_removed, idx_contract) = contract
    naive_cost = _flop_count(idx_contract, idx_removed, len(input_sets), idx_dict)
    comb_iter = itertools.combinations(range(len(input_sets)), 2)
    known_contractions = []
    path_cost = 0
    path = []
    for iteration in range(len(input_sets) - 1):
        for positions in comb_iter:
            if input_sets[positions[0]].isdisjoint(input_sets[positions[1]]):
                continue
            result = _parse_possible_contraction(positions, input_sets, output_set, idx_dict, memory_limit, path_cost, naive_cost)
            if result is not None:
                known_contractions.append(result)
        if len(known_contractions) == 0:
            for positions in itertools.combinations(range(len(input_sets)), 2):
                result = _parse_possible_contraction(positions, input_sets, output_set, idx_dict, memory_limit, path_cost, naive_cost)
                if result is not None:
                    known_contractions.append(result)
            if len(known_contractions) == 0:
                path.append(tuple(range(len(input_sets))))
                break
        best = min(known_contractions, key=lambda x: x[0])
        known_contractions = _update_other_results(known_contractions, best)
        input_sets = best[2]
        new_tensor_pos = len(input_sets) - 1
        comb_iter = ((i, new_tensor_pos) for i in range(new_tensor_pos))
        path.append(best[1])
        path_cost += best[0][1]
    return path