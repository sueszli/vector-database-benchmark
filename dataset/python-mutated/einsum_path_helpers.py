from itertools import combinations
from ivy.utils.einsum_parser import possibly_convert_to_numpy, convert_interleaved_input
einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
einsum_symbols_set = set(einsum_symbols)

def flop_count(idx_contraction, inner, num_terms, size_dictionary):
    if False:
        while True:
            i = 10
    "\n    Compute the number of FLOPS in the contraction.\n\n    Parameters\n    ----------\n    idx_contraction : iterable\n        The indices involved in the contraction\n    inner : bool\n        Does this contraction require an inner product?\n    num_terms : int\n        The number of terms in a contraction\n    size_dictionary : dict\n        The size of each of the indices in idx_contraction\n\n    Returns\n    -------\n    flop_count : int\n        The total number of FLOPS required for the contraction.\n\n    Examples\n    --------\n    >>> flop_count('abc', False, 1, {'a': 2, 'b':3, 'c':5})\n    30\n\n    >>> flop_count('abc', True, 2, {'a': 2, 'b':3, 'c':5})\n    60\n    "
    overall_size = compute_size_by_dict(idx_contraction, size_dictionary)
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1
    return overall_size * op_factor

def compute_size_by_dict(indices, idx_dict):
    if False:
        for i in range(10):
            print('nop')
    "\n    Compute the product of the elements in indices based on the dictionary idx_dict.\n\n    Parameters\n    ----------\n    indices : iterable\n        Indices to base the product on.\n    idx_dict : dictionary\n        Dictionary of index sizes\n\n    Returns\n    -------\n    ret : int\n        The resulting product.\n\n    Examples\n    --------\n    >>> compute_size_by_dict('abbc', {'a': 2, 'b':3, 'c':5})\n    90\n    "
    ret = 1
    for i in indices:
        ret *= idx_dict[i]
    return ret

def find_contraction(positions, input_sets, output_set):
    if False:
        return 10
    "\n    Find the contraction for a given set of input and output sets.\n\n    Parameters\n    ----------\n    positions : iterable\n        Integer positions of terms used in the contraction.\n    input_sets : list\n        List of sets that represent the lhs side of the einsum subscript\n    output_set : set\n        Set that represents the rhs side of the overall einsum subscript\n\n    Returns\n    -------\n    new_result : set\n        The indices of the resulting contraction\n    remaining : list\n        List of sets that have not been contracted, the new set is appended to\n        the end of this list\n    idx_removed : set\n        Indices removed from the entire contraction\n    idx_contraction : set\n        The indices used in the current contraction\n\n    Examples\n    --------\n    # A simple dot product test case\n    >>> pos = (0, 1)\n    >>> isets = [set('ab'), set('bc')]\n    >>> oset = set('ac')\n    >>> find_contraction(pos, isets, oset)\n    ({'a', 'c'}, [{'a', 'c'}], {'b'}, {'a', 'b', 'c'})\n    # A more complex case with additional terms in the contraction\n    >>> pos = (0, 2)\n    >>> isets = [set('abd'), set('ac'), set('bdc')]\n    >>> oset = set('ac')\n    >>> find_contraction(pos, isets, oset)\n    ({'a', 'c'}, [{'a', 'c'}, {'a', 'c'}], {'b', 'd'}, {'a', 'b', 'c', 'd'})\n    "
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

def optimal_path(input_sets, output_set, idx_dict, memory_limit):
    if False:
        i = 10
        return i + 15
    "\n    Compute all possible pair contractions, sieves the results based on ``memory_limit``\n    and returns the lowest cost path. This algorithm scales factorial with respect to\n    the elements in the list ``input_sets``.\n\n    Parameters\n    ----------\n    input_sets : list\n        List of sets that represent the lhs side of the einsum subscript\n    output_set : set\n        Set that represents the rhs side of the overall einsum subscript\n    idx_dict : dictionary\n        Dictionary of index sizes\n    memory_limit : int\n        The maximum number of elements in a temporary array\n\n    Returns\n    -------\n    path : list\n        The optimal contraction order within the memory limit constraint.\n\n    Examples\n    --------\n    >>> isets = [set('abd'), set('ac'), set('bdc')]\n    >>> oset = set()\n    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}\n    >>> optimal_path(isets, oset, idx_sizes, 5000)\n    [(0, 2), (0, 1)]\n    "
    full_results = [(0, [], input_sets)]
    for iteration in range(len(input_sets) - 1):
        iter_results = []
        for curr in full_results:
            (cost, positions, remaining) = curr
            for con in combinations(range(len(input_sets) - iteration), 2):
                cont = find_contraction(con, remaining, output_set)
                (new_result, new_input_sets, idx_removed, idx_contract) = cont
                new_size = compute_size_by_dict(new_result, idx_dict)
                if new_size > memory_limit:
                    continue
                total_cost = cost + flop_count(idx_contract, idx_removed, len(con), idx_dict)
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

def parse_possible_contraction(positions, input_sets, output_set, idx_dict, memory_limit, path_cost, naive_cost):
    if False:
        return 10
    '\n    Compute the cost (removed size + flops) and resultant indices for performing the\n    contraction specified by ``positions``.\n\n    Parameters\n    ----------\n    positions : tuple of int\n        The locations of the proposed tensors to contract.\n    input_sets : list of sets\n        The indices found on each tensors.\n    output_set : set\n        The output indices of the expression.\n    idx_dict : dict\n        Mapping of each index to its size.\n    memory_limit : int\n        The total allowed size for an intermediary tensor.\n    path_cost : int\n        The contraction cost so far.\n    naive_cost : int\n        The cost of the unoptimized expression.\n\n    Returns\n    -------\n    cost : (int, int)\n        A tuple containing the size of any indices removed, and the flop cost.\n    positions : tuple of int\n        The locations of the proposed tensors to contract.\n    new_input_sets : list of sets\n        The resulting new list of indices if this proposed contraction is performed.\n    '
    contract = find_contraction(positions, input_sets, output_set)
    (idx_result, new_input_sets, idx_removed, idx_contract) = contract
    new_size = compute_size_by_dict(idx_result, idx_dict)
    if new_size > memory_limit:
        return None
    old_sizes = (compute_size_by_dict(input_sets[p], idx_dict) for p in positions)
    removed_size = sum(old_sizes) - new_size
    cost = flop_count(idx_contract, idx_removed, len(positions), idx_dict)
    sort = (-removed_size, cost)
    if path_cost + cost > naive_cost:
        return None
    return [sort, positions, new_input_sets]

def update_other_results(results, best):
    if False:
        while True:
            i = 10
    '\n    Update the positions and provisional input_sets of ``results`` based on performing\n    the contraction result ``best``. Remove any involving the tensors contracted.\n\n    Parameters\n    ----------\n    results : list\n        List of contraction results produced by ``_parse_possible_contraction``.\n    best : list\n        The best contraction of ``results`` i.e. the one that will be performed.\n\n    Returns\n    -------\n    mod_results : list\n        The list of modified results, updated with outcome of ``best`` contraction.\n    '
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

def greedy_path(input_sets, output_set, idx_dict, memory_limit):
    if False:
        return 10
    "\n    Find the path by contracting the best pair until the input list is exhausted. The\n    best pair is found by minimizing the tuple ``(-prod(indices_removed), cost)``.  What\n    this amounts to is prioritizing matrix multiplication or inner product operations,\n    then Hadamard like operations, and finally outer operations. Outer products are\n    limited by ``memory_limit``. This algorithm scales cubically with respect to the\n    number of elements in the list ``input_sets``.\n\n    Parameters\n    ----------\n    input_sets : list\n        List of sets that represent the lhs side of the einsum subscript\n    output_set : set\n        Set that represents the rhs side of the overall einsum subscript\n    idx_dict : dictionary\n        Dictionary of index sizes\n    memory_limit : int\n        The maximum number of elements in a temporary array\n\n    Returns\n    -------\n    path : list\n        The greedy contraction order within the memory limit constraint.\n\n    Examples\n    --------\n    >>> isets = [set('abd'), set('ac'), set('bdc')]\n    >>> oset = set()\n    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}\n    >>> greedy_path(isets, oset, idx_sizes, 5000)\n    [(0, 2), (0, 1)]\n    "
    if len(input_sets) == 1:
        return [(0,)]
    elif len(input_sets) == 2:
        return [(0, 1)]
    contract = find_contraction(range(len(input_sets)), input_sets, output_set)
    (idx_result, new_input_sets, idx_removed, idx_contract) = contract
    naive_cost = flop_count(idx_contract, idx_removed, len(input_sets), idx_dict)
    comb_iter = combinations(range(len(input_sets)), 2)
    known_contractions = []
    path_cost = 0
    path = []
    for iteration in range(len(input_sets) - 1):
        for positions in comb_iter:
            if input_sets[positions[0]].isdisjoint(input_sets[positions[1]]):
                continue
            result = parse_possible_contraction(positions, input_sets, output_set, idx_dict, memory_limit, path_cost, naive_cost)
            if result is not None:
                known_contractions.append(result)
        if len(known_contractions) == 0:
            for positions in combinations(range(len(input_sets)), 2):
                result = parse_possible_contraction(positions, input_sets, output_set, idx_dict, memory_limit, path_cost, naive_cost)
                if result is not None:
                    known_contractions.append(result)
            if len(known_contractions) == 0:
                path.append(tuple(range(len(input_sets))))
                break
        best = min(known_contractions, key=lambda x: x[0])
        known_contractions = update_other_results(known_contractions, best)
        input_sets = best[2]
        new_tensor_pos = len(input_sets) - 1
        comb_iter = ((i, new_tensor_pos) for i in range(new_tensor_pos))
        path.append(best[1])
        path_cost += best[0][1]
    return path

def can_dot(inputs, result, idx_removed):
    if False:
        return 10
    "\n    Check if we can use BLAS (np.tensordot) call and its beneficial to do so.\n\n    Parameters\n    ----------\n    inputs : list of str\n        Specifies the subscripts for summation.\n    result : str\n        Resulting summation.\n    idx_removed : set\n        Indices that are removed in the summation\n\n    Returns\n    -------\n    type : bool\n        Returns true if BLAS should and can be used, else False\n\n    Notes\n    -----\n    If the operations is BLAS level 1 or 2 and is not already aligned\n    we default back to einsum as the memory movement to copy is more\n    costly than the operation itself.\n\n    Examples\n    --------\n    # Standard GEMM operation\n    >>> can_dot(['ij', 'jk'], 'ik', set('j'))\n    True\n    # Can use the standard BLAS, but requires odd data movement\n    >>> can_dot(['ijj', 'jk'], 'ik', set('j'))\n    False\n    # DDOT where the memory is not aligned\n    >>> can_dot(['ijk', 'ikj'], '', set('ijk'))\n    False\n    "
    if len(idx_removed) == 0:
        return False
    if len(inputs) != 2:
        return False
    (input_left, input_right) = inputs
    for c in set(input_left + input_right):
        (nl, nr) = (input_left.count(c), input_right.count(c))
        if nl > 1 or nr > 1 or nl + nr > 2:
            return False
        if nl + nr - 1 == int(c in result):
            return False
    set_left = set(input_left)
    set_right = set(input_right)
    keep_left = set_left - idx_removed
    keep_right = set_right - idx_removed
    rs = len(idx_removed)
    if input_left == input_right:
        return True
    if set_left == set_right:
        return False
    if input_left[-rs:] == input_right[:rs]:
        return True
    if input_left[:rs] == input_right[-rs:]:
        return True
    if input_left[-rs:] == input_right[-rs:]:
        return True
    if input_left[:rs] == input_right[:rs]:
        return True
    if not keep_left or not keep_right:
        return False
    return True

def parse_einsum_input(operands, subscripts=None):
    if False:
        return 10
    "\n    Reproduction of einsum c side einsum parsing in python.\n\n    Returns\n    -------\n    input_strings : str\n        Parsed input strings\n    output_string : str\n        Parsed output string\n    operands : list of array_like\n        The operands to use in the numpy contraction\n\n    Examples\n    --------\n    The operand list is simplified to reduce printing:\n\n    >>> np.random.seed(123)\n    >>> a = np.random.rand(4, 4)\n    >>> b = np.random.rand(4, 4, 4)\n    >>> parse_einsum_input(('...a,...a->...', a, b))\n    ('za,xza', 'xz', [a, b]) # may vary\n    >>> parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))\n    ('za,xza', 'xz', [a, b]) # may vary\n    "
    if len(operands) == 0:
        raise ValueError('No input operands')
    if subscripts:
        subscripts = subscripts.replace(' ', '')
        operands = [possibly_convert_to_numpy(x) for x in operands]
    elif isinstance(operands[0], str):
        subscripts = operands[0].replace(' ', '')
        operands = [possibly_convert_to_numpy(x) for x in operands[1:]]
    else:
        (subscripts, operands) = convert_interleaved_input(operands)
    if '-' in subscripts or '>' in subscripts:
        invalid = subscripts.count('-') > 1 or subscripts.count('>') > 1
        if invalid or subscripts.count('->') != 1:
            raise ValueError("Subscripts can only contain one '->'.")
    if '.' in subscripts:
        used = subscripts.replace('.', '').replace(',', '').replace('->', '')
        unused = list(einsum_symbols_set - set(used))
        ellipse_inds = ''.join(unused)
        longest = 0
        if '->' in subscripts:
            (input_tmp, output_sub) = subscripts.split('->')
            split_subscripts = input_tmp.split(',')
            out_sub = True
        else:
            split_subscripts = subscripts.split(',')
            out_sub = False
        for (num, sub) in enumerate(split_subscripts):
            if '.' in sub:
                if sub.count('.') != 3 or sub.count('...') != 1:
                    raise ValueError('Invalid Ellipses.')
                if operands[num].shape == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(operands[num].ndim, 1)
                    ellipse_count -= len(sub) - 3
                if ellipse_count > longest:
                    longest = ellipse_count
                if ellipse_count < 0:
                    raise ValueError('Ellipses lengths do not match.')
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace('...', '')
                else:
                    rep_inds = ellipse_inds[-ellipse_count:]
                    split_subscripts[num] = sub.replace('...', rep_inds)
        subscripts = ','.join(split_subscripts)
        if longest == 0:
            out_ellipse = ''
        else:
            out_ellipse = ellipse_inds[-longest:]
        if out_sub:
            subscripts += '->' + output_sub.replace('...', out_ellipse)
        else:
            output_subscript = ''
            tmp_subscripts = subscripts.replace(',', '')
            for s in sorted(set(tmp_subscripts)):
                if s not in einsum_symbols:
                    raise ValueError('Character %s is not a valid symbol.' % s)
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s
            normal_inds = ''.join(sorted(set(output_subscript) - set(out_ellipse)))
            subscripts += '->' + out_ellipse + normal_inds
    if '->' in subscripts:
        (input_subscripts, output_subscript) = subscripts.split('->')
    else:
        input_subscripts = subscripts
        tmp_subscripts = subscripts.replace(',', '')
        output_subscript = ''
        for s in sorted(set(tmp_subscripts)):
            if s not in einsum_symbols:
                raise ValueError('Character %s is not a valid symbol.' % s)
            if tmp_subscripts.count(s) == 1:
                output_subscript += s
    for char in output_subscript:
        if char not in input_subscripts:
            raise ValueError('Output character %s did not appear in the input' % char)
    if len(input_subscripts.split(',')) != len(operands):
        raise ValueError('Number of einsum subscripts must be equal to the number of operands.')
    return (input_subscripts, output_subscript, operands)