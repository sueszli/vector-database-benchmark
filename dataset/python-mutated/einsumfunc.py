"""
Implementation of optimized einsum.

"""
import itertools
import operator
from numpy._core.multiarray import c_einsum
from numpy._core.numeric import asanyarray, tensordot
from numpy._core.overrides import array_function_dispatch
__all__ = ['einsum', 'einsum_path']
einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
einsum_symbols_set = set(einsum_symbols)

def _flop_count(idx_contraction, inner, num_terms, size_dictionary):
    if False:
        while True:
            i = 10
    "\n    Computes the number of FLOPS in the contraction.\n\n    Parameters\n    ----------\n    idx_contraction : iterable\n        The indices involved in the contraction\n    inner : bool\n        Does this contraction require an inner product?\n    num_terms : int\n        The number of terms in a contraction\n    size_dictionary : dict\n        The size of each of the indices in idx_contraction\n\n    Returns\n    -------\n    flop_count : int\n        The total number of FLOPS required for the contraction.\n\n    Examples\n    --------\n\n    >>> _flop_count('abc', False, 1, {'a': 2, 'b':3, 'c':5})\n    30\n\n    >>> _flop_count('abc', True, 2, {'a': 2, 'b':3, 'c':5})\n    60\n\n    "
    overall_size = _compute_size_by_dict(idx_contraction, size_dictionary)
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1
    return overall_size * op_factor

def _compute_size_by_dict(indices, idx_dict):
    if False:
        print('Hello World!')
    "\n    Computes the product of the elements in indices based on the dictionary\n    idx_dict.\n\n    Parameters\n    ----------\n    indices : iterable\n        Indices to base the product on.\n    idx_dict : dictionary\n        Dictionary of index sizes\n\n    Returns\n    -------\n    ret : int\n        The resulting product.\n\n    Examples\n    --------\n    >>> _compute_size_by_dict('abbc', {'a': 2, 'b':3, 'c':5})\n    90\n\n    "
    ret = 1
    for i in indices:
        ret *= idx_dict[i]
    return ret

def _find_contraction(positions, input_sets, output_set):
    if False:
        while True:
            i = 10
    "\n    Finds the contraction for a given set of input and output sets.\n\n    Parameters\n    ----------\n    positions : iterable\n        Integer positions of terms used in the contraction.\n    input_sets : list\n        List of sets that represent the lhs side of the einsum subscript\n    output_set : set\n        Set that represents the rhs side of the overall einsum subscript\n\n    Returns\n    -------\n    new_result : set\n        The indices of the resulting contraction\n    remaining : list\n        List of sets that have not been contracted, the new set is appended to\n        the end of this list\n    idx_removed : set\n        Indices removed from the entire contraction\n    idx_contraction : set\n        The indices used in the current contraction\n\n    Examples\n    --------\n\n    # A simple dot product test case\n    >>> pos = (0, 1)\n    >>> isets = [set('ab'), set('bc')]\n    >>> oset = set('ac')\n    >>> _find_contraction(pos, isets, oset)\n    ({'a', 'c'}, [{'a', 'c'}], {'b'}, {'a', 'b', 'c'})\n\n    # A more complex case with additional terms in the contraction\n    >>> pos = (0, 2)\n    >>> isets = [set('abd'), set('ac'), set('bdc')]\n    >>> oset = set('ac')\n    >>> _find_contraction(pos, isets, oset)\n    ({'a', 'c'}, [{'a', 'c'}, {'a', 'c'}], {'b', 'd'}, {'a', 'b', 'c', 'd'})\n    "
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
        for i in range(10):
            print('nop')
    "\n    Computes all possible pair contractions, sieves the results based\n    on ``memory_limit`` and returns the lowest cost path. This algorithm\n    scales factorial with respect to the elements in the list ``input_sets``.\n\n    Parameters\n    ----------\n    input_sets : list\n        List of sets that represent the lhs side of the einsum subscript\n    output_set : set\n        Set that represents the rhs side of the overall einsum subscript\n    idx_dict : dictionary\n        Dictionary of index sizes\n    memory_limit : int\n        The maximum number of elements in a temporary array\n\n    Returns\n    -------\n    path : list\n        The optimal contraction order within the memory limit constraint.\n\n    Examples\n    --------\n    >>> isets = [set('abd'), set('ac'), set('bdc')]\n    >>> oset = set()\n    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}\n    >>> _optimal_path(isets, oset, idx_sizes, 5000)\n    [(0, 2), (0, 1)]\n    "
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
        for i in range(10):
            print('nop')
    'Compute the cost (removed size + flops) and resultant indices for\n    performing the contraction specified by ``positions``.\n\n    Parameters\n    ----------\n    positions : tuple of int\n        The locations of the proposed tensors to contract.\n    input_sets : list of sets\n        The indices found on each tensors.\n    output_set : set\n        The output indices of the expression.\n    idx_dict : dict\n        Mapping of each index to its size.\n    memory_limit : int\n        The total allowed size for an intermediary tensor.\n    path_cost : int\n        The contraction cost so far.\n    naive_cost : int\n        The cost of the unoptimized expression.\n\n    Returns\n    -------\n    cost : (int, int)\n        A tuple containing the size of any indices removed, and the flop cost.\n    positions : tuple of int\n        The locations of the proposed tensors to contract.\n    new_input_sets : list of sets\n        The resulting new list of indices if this proposed contraction\n        is performed.\n\n    '
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
        for i in range(10):
            print('nop')
    'Update the positions and provisional input_sets of ``results``\n    based on performing the contraction result ``best``. Remove any\n    involving the tensors contracted.\n\n    Parameters\n    ----------\n    results : list\n        List of contraction results produced by \n        ``_parse_possible_contraction``.\n    best : list\n        The best contraction of ``results`` i.e. the one that\n        will be performed.\n\n    Returns\n    -------\n    mod_results : list\n        The list of modified results, updated with outcome of\n        ``best`` contraction.\n    '
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
        for i in range(10):
            print('nop')
    "\n    Finds the path by contracting the best pair until the input list is\n    exhausted. The best pair is found by minimizing the tuple\n    ``(-prod(indices_removed), cost)``.  What this amounts to is prioritizing\n    matrix multiplication or inner product operations, then Hadamard like\n    operations, and finally outer operations. Outer products are limited by\n    ``memory_limit``. This algorithm scales cubically with respect to the\n    number of elements in the list ``input_sets``.\n\n    Parameters\n    ----------\n    input_sets : list\n        List of sets that represent the lhs side of the einsum subscript\n    output_set : set\n        Set that represents the rhs side of the overall einsum subscript\n    idx_dict : dictionary\n        Dictionary of index sizes\n    memory_limit : int\n        The maximum number of elements in a temporary array\n\n    Returns\n    -------\n    path : list\n        The greedy contraction order within the memory limit constraint.\n\n    Examples\n    --------\n    >>> isets = [set('abd'), set('ac'), set('bdc')]\n    >>> oset = set()\n    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}\n    >>> _greedy_path(isets, oset, idx_sizes, 5000)\n    [(0, 2), (0, 1)]\n    "
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

def _can_dot(inputs, result, idx_removed):
    if False:
        return 10
    "\n    Checks if we can use BLAS (np.tensordot) call and its beneficial to do so.\n\n    Parameters\n    ----------\n    inputs : list of str\n        Specifies the subscripts for summation.\n    result : str\n        Resulting summation.\n    idx_removed : set\n        Indices that are removed in the summation\n\n\n    Returns\n    -------\n    type : bool\n        Returns true if BLAS should and can be used, else False\n\n    Notes\n    -----\n    If the operations is BLAS level 1 or 2 and is not already aligned\n    we default back to einsum as the memory movement to copy is more\n    costly than the operation itself.\n\n\n    Examples\n    --------\n\n    # Standard GEMM operation\n    >>> _can_dot(['ij', 'jk'], 'ik', set('j'))\n    True\n\n    # Can use the standard BLAS, but requires odd data movement\n    >>> _can_dot(['ijj', 'jk'], 'ik', set('j'))\n    False\n\n    # DDOT where the memory is not aligned\n    >>> _can_dot(['ijk', 'ikj'], '', set('ijk'))\n    False\n\n    "
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

def _parse_einsum_input(operands):
    if False:
        while True:
            i = 10
    "\n    A reproduction of einsum c side einsum parsing in python.\n\n    Returns\n    -------\n    input_strings : str\n        Parsed input strings\n    output_string : str\n        Parsed output string\n    operands : list of array_like\n        The operands to use in the numpy contraction\n\n    Examples\n    --------\n    The operand list is simplified to reduce printing:\n\n    >>> np.random.seed(123)\n    >>> a = np.random.rand(4, 4)\n    >>> b = np.random.rand(4, 4, 4)\n    >>> _parse_einsum_input(('...a,...a->...', a, b))\n    ('za,xza', 'xz', [a, b]) # may vary\n\n    >>> _parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))\n    ('za,xza', 'xz', [a, b]) # may vary\n    "
    if len(operands) == 0:
        raise ValueError('No input operands')
    if isinstance(operands[0], str):
        subscripts = operands[0].replace(' ', '')
        operands = [asanyarray(v) for v in operands[1:]]
        for s in subscripts:
            if s in '.,->':
                continue
            if s not in einsum_symbols:
                raise ValueError('Character %s is not a valid symbol.' % s)
    else:
        tmp_operands = list(operands)
        operand_list = []
        subscript_list = []
        for p in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))
        output_list = tmp_operands[-1] if len(tmp_operands) else None
        operands = [asanyarray(v) for v in operand_list]
        subscripts = ''
        last = len(subscript_list) - 1
        for (num, sub) in enumerate(subscript_list):
            for s in sub:
                if s is Ellipsis:
                    subscripts += '...'
                else:
                    try:
                        s = operator.index(s)
                    except TypeError as e:
                        raise TypeError('For this input type lists must contain either int or Ellipsis') from e
                    subscripts += einsum_symbols[s]
            if num != last:
                subscripts += ','
        if output_list is not None:
            subscripts += '->'
            for s in output_list:
                if s is Ellipsis:
                    subscripts += '...'
                else:
                    try:
                        s = operator.index(s)
                    except TypeError as e:
                        raise TypeError('For this input type lists must contain either int or Ellipsis') from e
                    subscripts += einsum_symbols[s]
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

def _einsum_path_dispatcher(*operands, optimize=None, einsum_call=None):
    if False:
        while True:
            i = 10
    return operands

@array_function_dispatch(_einsum_path_dispatcher, module='numpy')
def einsum_path(*operands, optimize='greedy', einsum_call=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    einsum_path(subscripts, *operands, optimize='greedy')\n\n    Evaluates the lowest cost contraction order for an einsum expression by\n    considering the creation of intermediate arrays.\n\n    Parameters\n    ----------\n    subscripts : str\n        Specifies the subscripts for summation.\n    *operands : list of array_like\n        These are the arrays for the operation.\n    optimize : {bool, list, tuple, 'greedy', 'optimal'}\n        Choose the type of path. If a tuple is provided, the second argument is\n        assumed to be the maximum intermediate size created. If only a single\n        argument is provided the largest input or output array size is used\n        as a maximum intermediate size.\n\n        * if a list is given that starts with ``einsum_path``, uses this as the\n          contraction path\n        * if False no optimization is taken\n        * if True defaults to the 'greedy' algorithm\n        * 'optimal' An algorithm that combinatorially explores all possible\n          ways of contracting the listed tensors and chooses the least costly\n          path. Scales exponentially with the number of terms in the\n          contraction.\n        * 'greedy' An algorithm that chooses the best pair contraction\n          at each step. Effectively, this algorithm searches the largest inner,\n          Hadamard, and then outer products at each step. Scales cubically with\n          the number of terms in the contraction. Equivalent to the 'optimal'\n          path for most contractions.\n\n        Default is 'greedy'.\n\n    Returns\n    -------\n    path : list of tuples\n        A list representation of the einsum path.\n    string_repr : str\n        A printable representation of the einsum path.\n\n    Notes\n    -----\n    The resulting path indicates which terms of the input contraction should be\n    contracted first, the result of this contraction is then appended to the\n    end of the contraction list. This list can then be iterated over until all\n    intermediate contractions are complete.\n\n    See Also\n    --------\n    einsum, linalg.multi_dot\n\n    Examples\n    --------\n\n    We can begin with a chain dot example. In this case, it is optimal to\n    contract the ``b`` and ``c`` tensors first as represented by the first\n    element of the path ``(1, 2)``. The resulting tensor is added to the end\n    of the contraction and the remaining contraction ``(0, 1)`` is then\n    completed.\n\n    >>> np.random.seed(123)\n    >>> a = np.random.rand(2, 2)\n    >>> b = np.random.rand(2, 5)\n    >>> c = np.random.rand(5, 2)\n    >>> path_info = np.einsum_path('ij,jk,kl->il', a, b, c, optimize='greedy')\n    >>> print(path_info[0])\n    ['einsum_path', (1, 2), (0, 1)]\n    >>> print(path_info[1])\n      Complete contraction:  ij,jk,kl->il # may vary\n             Naive scaling:  4\n         Optimized scaling:  3\n          Naive FLOP count:  1.600e+02\n      Optimized FLOP count:  5.600e+01\n       Theoretical speedup:  2.857\n      Largest intermediate:  4.000e+00 elements\n    -------------------------------------------------------------------------\n    scaling                  current                                remaining\n    -------------------------------------------------------------------------\n       3                   kl,jk->jl                                ij,jl->il\n       3                   jl,ij->il                                   il->il\n\n\n    A more complex index transformation example.\n\n    >>> I = np.random.rand(10, 10, 10, 10)\n    >>> C = np.random.rand(10, 10)\n    >>> path_info = np.einsum_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C,\n    ...                            optimize='greedy')\n\n    >>> print(path_info[0])\n    ['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)]\n    >>> print(path_info[1]) \n      Complete contraction:  ea,fb,abcd,gc,hd->efgh # may vary\n             Naive scaling:  8\n         Optimized scaling:  5\n          Naive FLOP count:  8.000e+08\n      Optimized FLOP count:  8.000e+05\n       Theoretical speedup:  1000.000\n      Largest intermediate:  1.000e+04 elements\n    --------------------------------------------------------------------------\n    scaling                  current                                remaining\n    --------------------------------------------------------------------------\n       5               abcd,ea->bcde                      fb,gc,hd,bcde->efgh\n       5               bcde,fb->cdef                         gc,hd,cdef->efgh\n       5               cdef,gc->defg                            hd,defg->efgh\n       5               defg,hd->efgh                               efgh->efgh\n    "
    path_type = optimize
    if path_type is True:
        path_type = 'greedy'
    if path_type is None:
        path_type = False
    explicit_einsum_path = False
    memory_limit = None
    if path_type is False or isinstance(path_type, str):
        pass
    elif len(path_type) and path_type[0] == 'einsum_path':
        explicit_einsum_path = True
    elif len(path_type) == 2 and isinstance(path_type[0], str) and isinstance(path_type[1], (int, float)):
        memory_limit = int(path_type[1])
        path_type = path_type[0]
    else:
        raise TypeError('Did not understand the path: %s' % str(path_type))
    einsum_call_arg = einsum_call
    (input_subscripts, output_subscript, operands) = _parse_einsum_input(operands)
    input_list = input_subscripts.split(',')
    input_sets = [set(x) for x in input_list]
    output_set = set(output_subscript)
    indices = set(input_subscripts.replace(',', ''))
    dimension_dict = {}
    broadcast_indices = [[] for x in range(len(input_list))]
    for (tnum, term) in enumerate(input_list):
        sh = operands[tnum].shape
        if len(sh) != len(term):
            raise ValueError('Einstein sum subscript %s does not contain the correct number of indices for operand %d.' % (input_subscripts[tnum], tnum))
        for (cnum, char) in enumerate(term):
            dim = sh[cnum]
            if dim == 1:
                broadcast_indices[tnum].append(char)
            if char in dimension_dict.keys():
                if dimension_dict[char] == 1:
                    dimension_dict[char] = dim
                elif dim not in (1, dimension_dict[char]):
                    raise ValueError("Size of label '%s' for operand %d (%d) does not match previous terms (%d)." % (char, tnum, dimension_dict[char], dim))
            else:
                dimension_dict[char] = dim
    broadcast_indices = [set(x) for x in broadcast_indices]
    size_list = [_compute_size_by_dict(term, dimension_dict) for term in input_list + [output_subscript]]
    max_size = max(size_list)
    if memory_limit is None:
        memory_arg = max_size
    else:
        memory_arg = memory_limit
    inner_product = sum((len(x) for x in input_sets)) - len(indices) > 0
    naive_cost = _flop_count(indices, inner_product, len(input_list), dimension_dict)
    if explicit_einsum_path:
        path = path_type[1:]
    elif path_type is False or len(input_list) in [1, 2] or indices == output_set:
        path = [tuple(range(len(input_list)))]
    elif path_type == 'greedy':
        path = _greedy_path(input_sets, output_set, dimension_dict, memory_arg)
    elif path_type == 'optimal':
        path = _optimal_path(input_sets, output_set, dimension_dict, memory_arg)
    else:
        raise KeyError('Path name %s not found', path_type)
    (cost_list, scale_list, size_list, contraction_list) = ([], [], [], [])
    for (cnum, contract_inds) in enumerate(path):
        contract_inds = tuple(sorted(list(contract_inds), reverse=True))
        contract = _find_contraction(contract_inds, input_sets, output_set)
        (out_inds, input_sets, idx_removed, idx_contract) = contract
        cost = _flop_count(idx_contract, idx_removed, len(contract_inds), dimension_dict)
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(_compute_size_by_dict(out_inds, dimension_dict))
        bcast = set()
        tmp_inputs = []
        for x in contract_inds:
            tmp_inputs.append(input_list.pop(x))
            bcast |= broadcast_indices.pop(x)
        new_bcast_inds = bcast - idx_removed
        if not len(idx_removed & bcast):
            do_blas = _can_dot(tmp_inputs, out_inds, idx_removed)
        else:
            do_blas = False
        if cnum - len(path) == -1:
            idx_result = output_subscript
        else:
            sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
            idx_result = ''.join([x[1] for x in sorted(sort_result)])
        input_list.append(idx_result)
        broadcast_indices.append(new_bcast_inds)
        einsum_str = ','.join(tmp_inputs) + '->' + idx_result
        contraction = (contract_inds, idx_removed, einsum_str, input_list[:], do_blas)
        contraction_list.append(contraction)
    opt_cost = sum(cost_list) + 1
    if len(input_list) != 1:
        raise RuntimeError('Invalid einsum_path is specified: {} more operands has to be contracted.'.format(len(input_list) - 1))
    if einsum_call_arg:
        return (operands, contraction_list)
    overall_contraction = input_subscripts + '->' + output_subscript
    header = ('scaling', 'current', 'remaining')
    speedup = naive_cost / opt_cost
    max_i = max(size_list)
    path_print = '  Complete contraction:  %s\n' % overall_contraction
    path_print += '         Naive scaling:  %d\n' % len(indices)
    path_print += '     Optimized scaling:  %d\n' % max(scale_list)
    path_print += '      Naive FLOP count:  %.3e\n' % naive_cost
    path_print += '  Optimized FLOP count:  %.3e\n' % opt_cost
    path_print += '   Theoretical speedup:  %3.3f\n' % speedup
    path_print += '  Largest intermediate:  %.3e elements\n' % max_i
    path_print += '-' * 74 + '\n'
    path_print += '%6s %24s %40s\n' % header
    path_print += '-' * 74
    for (n, contraction) in enumerate(contraction_list):
        (inds, idx_rm, einsum_str, remaining, blas) = contraction
        remaining_str = ','.join(remaining) + '->' + output_subscript
        path_run = (scale_list[n], einsum_str, remaining_str)
        path_print += '\n%4d    %24s %40s' % path_run
    path = ['einsum_path'] + path
    return (path, path_print)

def _einsum_dispatcher(*operands, out=None, optimize=None, **kwargs):
    if False:
        return 10
    yield from operands
    yield out

@array_function_dispatch(_einsum_dispatcher, module='numpy')
def einsum(*operands, out=None, optimize=False, **kwargs):
    if False:
        print('Hello World!')
    "\n    einsum(subscripts, *operands, out=None, dtype=None, order='K',\n           casting='safe', optimize=False)\n\n    Evaluates the Einstein summation convention on the operands.\n\n    Using the Einstein summation convention, many common multi-dimensional,\n    linear algebraic array operations can be represented in a simple fashion.\n    In *implicit* mode `einsum` computes these values.\n\n    In *explicit* mode, `einsum` provides further flexibility to compute\n    other array operations that might not be considered classical Einstein\n    summation operations, by disabling, or forcing summation over specified\n    subscript labels.\n\n    See the notes and examples for clarification.\n\n    Parameters\n    ----------\n    subscripts : str\n        Specifies the subscripts for summation as comma separated list of\n        subscript labels. An implicit (classical Einstein summation)\n        calculation is performed unless the explicit indicator '->' is\n        included as well as subscript labels of the precise output form.\n    operands : list of array_like\n        These are the arrays for the operation.\n    out : ndarray, optional\n        If provided, the calculation is done into this array.\n    dtype : {data-type, None}, optional\n        If provided, forces the calculation to use the data type specified.\n        Note that you may have to also give a more liberal `casting`\n        parameter to allow the conversions. Default is None.\n    order : {'C', 'F', 'A', 'K'}, optional\n        Controls the memory layout of the output. 'C' means it should\n        be C contiguous. 'F' means it should be Fortran contiguous,\n        'A' means it should be 'F' if the inputs are all 'F', 'C' otherwise.\n        'K' means it should be as close to the layout as the inputs as\n        is possible, including arbitrarily permuted axes.\n        Default is 'K'.\n    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional\n        Controls what kind of data casting may occur.  Setting this to\n        'unsafe' is not recommended, as it can adversely affect accumulations.\n\n        * 'no' means the data types should not be cast at all.\n        * 'equiv' means only byte-order changes are allowed.\n        * 'safe' means only casts which can preserve values are allowed.\n        * 'same_kind' means only safe casts or casts within a kind,\n          like float64 to float32, are allowed.\n        * 'unsafe' means any data conversions may be done.\n\n        Default is 'safe'.\n    optimize : {False, True, 'greedy', 'optimal'}, optional\n        Controls if intermediate optimization should occur. No optimization\n        will occur if False and True will default to the 'greedy' algorithm.\n        Also accepts an explicit contraction list from the ``np.einsum_path``\n        function. See ``np.einsum_path`` for more details. Defaults to False.\n\n    Returns\n    -------\n    output : ndarray\n        The calculation based on the Einstein summation convention.\n\n    See Also\n    --------\n    einsum_path, dot, inner, outer, tensordot, linalg.multi_dot\n    einsum:\n        Similar verbose interface is provided by the\n        `einops <https://github.com/arogozhnikov/einops>`_ package to cover\n        additional operations: transpose, reshape/flatten, repeat/tile,\n        squeeze/unsqueeze and reductions.\n        The `opt_einsum <https://optimized-einsum.readthedocs.io/en/stable/>`_\n        optimizes contraction order for einsum-like expressions\n        in backend-agnostic manner.\n\n    Notes\n    -----\n    .. versionadded:: 1.6.0\n\n    The Einstein summation convention can be used to compute\n    many multi-dimensional, linear algebraic array operations. `einsum`\n    provides a succinct way of representing these.\n\n    A non-exhaustive list of these operations,\n    which can be computed by `einsum`, is shown below along with examples:\n\n    * Trace of an array, :py:func:`numpy.trace`.\n    * Return a diagonal, :py:func:`numpy.diag`.\n    * Array axis summations, :py:func:`numpy.sum`.\n    * Transpositions and permutations, :py:func:`numpy.transpose`.\n    * Matrix multiplication and dot product, :py:func:`numpy.matmul`\n        :py:func:`numpy.dot`.\n    * Vector inner and outer products, :py:func:`numpy.inner`\n        :py:func:`numpy.outer`.\n    * Broadcasting, element-wise and scalar multiplication,\n        :py:func:`numpy.multiply`.\n    * Tensor contractions, :py:func:`numpy.tensordot`.\n    * Chained array operations, in efficient calculation order,\n        :py:func:`numpy.einsum_path`.\n\n    The subscripts string is a comma-separated list of subscript labels,\n    where each label refers to a dimension of the corresponding operand.\n    Whenever a label is repeated it is summed, so ``np.einsum('i,i', a, b)``\n    is equivalent to :py:func:`np.inner(a,b) <numpy.inner>`. If a label\n    appears only once, it is not summed, so ``np.einsum('i', a)``\n    produces a view of ``a`` with no changes. A further example\n    ``np.einsum('ij,jk', a, b)`` describes traditional matrix multiplication\n    and is equivalent to :py:func:`np.matmul(a,b) <numpy.matmul>`.\n    Repeated subscript labels in one operand take the diagonal.\n    For example, ``np.einsum('ii', a)`` is equivalent to\n    :py:func:`np.trace(a) <numpy.trace>`.\n\n    In *implicit mode*, the chosen subscripts are important\n    since the axes of the output are reordered alphabetically.  This\n    means that ``np.einsum('ij', a)`` doesn't affect a 2D array, while\n    ``np.einsum('ji', a)`` takes its transpose. Additionally,\n    ``np.einsum('ij,jk', a, b)`` returns a matrix multiplication, while,\n    ``np.einsum('ij,jh', a, b)`` returns the transpose of the\n    multiplication since subscript 'h' precedes subscript 'i'.\n\n    In *explicit mode* the output can be directly controlled by\n    specifying output subscript labels.  This requires the\n    identifier '->' as well as the list of output subscript labels.\n    This feature increases the flexibility of the function since\n    summing can be disabled or forced when required. The call\n    ``np.einsum('i->', a)`` is like\n    :py:func:`np.sum(a, axis=-1) <numpy.sum>`, and ``np.einsum('ii->i', a)``\n    is like :py:func:`np.diag(a) <numpy.diag>`. The difference is that\n    `einsum` does not allow broadcasting by default. Additionally\n    ``np.einsum('ij,jh->ih', a, b)`` directly specifies the order of\n    the output subscript labels and therefore returns matrix multiplication,\n    unlike the example above in implicit mode.\n\n    To enable and control broadcasting, use an ellipsis.  Default\n    NumPy-style broadcasting is done by adding an ellipsis\n    to the left of each term, like ``np.einsum('...ii->...i', a)``.\n    To take the trace along the first and last axes,\n    you can do ``np.einsum('i...i', a)``, or to do a matrix-matrix\n    product with the left-most indices instead of rightmost, one can do\n    ``np.einsum('ij...,jk...->ik...', a, b)``.\n\n    When there is only one operand, no axes are summed, and no output\n    parameter is provided, a view into the operand is returned instead\n    of a new array.  Thus, taking the diagonal as ``np.einsum('ii->i', a)``\n    produces a view (changed in version 1.10.0).\n\n    `einsum` also provides an alternative way to provide the subscripts and\n    operands as ``einsum(op0, sublist0, op1, sublist1, ..., [sublistout])``.\n    If the output shape is not provided in this format `einsum` will be\n    calculated in implicit mode, otherwise it will be performed explicitly.\n    The examples below have corresponding `einsum` calls with the two\n    parameter methods.\n\n    .. versionadded:: 1.10.0\n\n    Views returned from einsum are now writeable whenever the input array\n    is writeable. For example, ``np.einsum('ijk...->kji...', a)`` will now\n    have the same effect as :py:func:`np.swapaxes(a, 0, 2) <numpy.swapaxes>`\n    and ``np.einsum('ii->i', a)`` will return a writeable view of the diagonal\n    of a 2D array.\n\n    .. versionadded:: 1.12.0\n\n    Added the ``optimize`` argument which will optimize the contraction order\n    of an einsum expression. For a contraction with three or more operands\n    this can greatly increase the computational efficiency at the cost of\n    a larger memory footprint during computation.\n\n    Typically a 'greedy' algorithm is applied which empirical tests have shown\n    returns the optimal path in the majority of cases. In some cases 'optimal'\n    will return the superlative path through a more expensive, exhaustive\n    search. For iterative calculations it may be advisable to calculate\n    the optimal path once and reuse that path by supplying it as an argument.\n    An example is given below.\n\n    See :py:func:`numpy.einsum_path` for more details.\n\n    Examples\n    --------\n    >>> a = np.arange(25).reshape(5,5)\n    >>> b = np.arange(5)\n    >>> c = np.arange(6).reshape(2,3)\n\n    Trace of a matrix:\n\n    >>> np.einsum('ii', a)\n    60\n    >>> np.einsum(a, [0,0])\n    60\n    >>> np.trace(a)\n    60\n\n    Extract the diagonal (requires explicit form):\n\n    >>> np.einsum('ii->i', a)\n    array([ 0,  6, 12, 18, 24])\n    >>> np.einsum(a, [0,0], [0])\n    array([ 0,  6, 12, 18, 24])\n    >>> np.diag(a)\n    array([ 0,  6, 12, 18, 24])\n\n    Sum over an axis (requires explicit form):\n\n    >>> np.einsum('ij->i', a)\n    array([ 10,  35,  60,  85, 110])\n    >>> np.einsum(a, [0,1], [0])\n    array([ 10,  35,  60,  85, 110])\n    >>> np.sum(a, axis=1)\n    array([ 10,  35,  60,  85, 110])\n\n    For higher dimensional arrays summing a single axis can be done\n    with ellipsis:\n\n    >>> np.einsum('...j->...', a)\n    array([ 10,  35,  60,  85, 110])\n    >>> np.einsum(a, [Ellipsis,1], [Ellipsis])\n    array([ 10,  35,  60,  85, 110])\n\n    Compute a matrix transpose, or reorder any number of axes:\n\n    >>> np.einsum('ji', c)\n    array([[0, 3],\n           [1, 4],\n           [2, 5]])\n    >>> np.einsum('ij->ji', c)\n    array([[0, 3],\n           [1, 4],\n           [2, 5]])\n    >>> np.einsum(c, [1,0])\n    array([[0, 3],\n           [1, 4],\n           [2, 5]])\n    >>> np.transpose(c)\n    array([[0, 3],\n           [1, 4],\n           [2, 5]])\n\n    Vector inner products:\n\n    >>> np.einsum('i,i', b, b)\n    30\n    >>> np.einsum(b, [0], b, [0])\n    30\n    >>> np.inner(b,b)\n    30\n\n    Matrix vector multiplication:\n\n    >>> np.einsum('ij,j', a, b)\n    array([ 30,  80, 130, 180, 230])\n    >>> np.einsum(a, [0,1], b, [1])\n    array([ 30,  80, 130, 180, 230])\n    >>> np.dot(a, b)\n    array([ 30,  80, 130, 180, 230])\n    >>> np.einsum('...j,j', a, b)\n    array([ 30,  80, 130, 180, 230])\n\n    Broadcasting and scalar multiplication:\n\n    >>> np.einsum('..., ...', 3, c)\n    array([[ 0,  3,  6],\n           [ 9, 12, 15]])\n    >>> np.einsum(',ij', 3, c)\n    array([[ 0,  3,  6],\n           [ 9, 12, 15]])\n    >>> np.einsum(3, [Ellipsis], c, [Ellipsis])\n    array([[ 0,  3,  6],\n           [ 9, 12, 15]])\n    >>> np.multiply(3, c)\n    array([[ 0,  3,  6],\n           [ 9, 12, 15]])\n\n    Vector outer product:\n\n    >>> np.einsum('i,j', np.arange(2)+1, b)\n    array([[0, 1, 2, 3, 4],\n           [0, 2, 4, 6, 8]])\n    >>> np.einsum(np.arange(2)+1, [0], b, [1])\n    array([[0, 1, 2, 3, 4],\n           [0, 2, 4, 6, 8]])\n    >>> np.outer(np.arange(2)+1, b)\n    array([[0, 1, 2, 3, 4],\n           [0, 2, 4, 6, 8]])\n\n    Tensor contraction:\n\n    >>> a = np.arange(60.).reshape(3,4,5)\n    >>> b = np.arange(24.).reshape(4,3,2)\n    >>> np.einsum('ijk,jil->kl', a, b)\n    array([[4400., 4730.],\n           [4532., 4874.],\n           [4664., 5018.],\n           [4796., 5162.],\n           [4928., 5306.]])\n    >>> np.einsum(a, [0,1,2], b, [1,0,3], [2,3])\n    array([[4400., 4730.],\n           [4532., 4874.],\n           [4664., 5018.],\n           [4796., 5162.],\n           [4928., 5306.]])\n    >>> np.tensordot(a,b, axes=([1,0],[0,1]))\n    array([[4400., 4730.],\n           [4532., 4874.],\n           [4664., 5018.],\n           [4796., 5162.],\n           [4928., 5306.]])\n\n    Writeable returned arrays (since version 1.10.0):\n\n    >>> a = np.zeros((3, 3))\n    >>> np.einsum('ii->i', a)[:] = 1\n    >>> a\n    array([[1., 0., 0.],\n           [0., 1., 0.],\n           [0., 0., 1.]])\n\n    Example of ellipsis use:\n\n    >>> a = np.arange(6).reshape((3,2))\n    >>> b = np.arange(12).reshape((4,3))\n    >>> np.einsum('ki,jk->ij', a, b)\n    array([[10, 28, 46, 64],\n           [13, 40, 67, 94]])\n    >>> np.einsum('ki,...k->i...', a, b)\n    array([[10, 28, 46, 64],\n           [13, 40, 67, 94]])\n    >>> np.einsum('k...,jk', a, b)\n    array([[10, 28, 46, 64],\n           [13, 40, 67, 94]])\n\n    Chained array operations. For more complicated contractions, speed ups\n    might be achieved by repeatedly computing a 'greedy' path or pre-computing\n    the 'optimal' path and repeatedly applying it, using an `einsum_path`\n    insertion (since version 1.12.0). Performance improvements can be\n    particularly significant with larger arrays:\n\n    >>> a = np.ones(64).reshape(2,4,8)\n\n    Basic `einsum`: ~1520ms  (benchmarked on 3.1GHz Intel i5.)\n\n    >>> for iteration in range(500):\n    ...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a)\n\n    Sub-optimal `einsum` (due to repeated path calculation time): ~330ms\n\n    >>> for iteration in range(500):\n    ...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a,\n    ...         optimize='optimal')\n\n    Greedy `einsum` (faster optimal path approximation): ~160ms\n\n    >>> for iteration in range(500):\n    ...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize='greedy')\n\n    Optimal `einsum` (best usage pattern in some use cases): ~110ms\n\n    >>> path = np.einsum_path('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, \n    ...     optimize='optimal')[0]\n    >>> for iteration in range(500):\n    ...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize=path)\n\n    "
    specified_out = out is not None
    if optimize is False:
        if specified_out:
            kwargs['out'] = out
        return c_einsum(*operands, **kwargs)
    valid_einsum_kwargs = ['dtype', 'order', 'casting']
    unknown_kwargs = [k for (k, v) in kwargs.items() if k not in valid_einsum_kwargs]
    if len(unknown_kwargs):
        raise TypeError('Did not understand the following kwargs: %s' % unknown_kwargs)
    (operands, contraction_list) = einsum_path(*operands, optimize=optimize, einsum_call=True)
    output_order = kwargs.pop('order', 'K')
    if output_order.upper() == 'A':
        if all((arr.flags.f_contiguous for arr in operands)):
            output_order = 'F'
        else:
            output_order = 'C'
    for (num, contraction) in enumerate(contraction_list):
        (inds, idx_rm, einsum_str, remaining, blas) = contraction
        tmp_operands = [operands.pop(x) for x in inds]
        handle_out = specified_out and num + 1 == len(contraction_list)
        if blas:
            (input_str, results_index) = einsum_str.split('->')
            (input_left, input_right) = input_str.split(',')
            tensor_result = input_left + input_right
            for s in idx_rm:
                tensor_result = tensor_result.replace(s, '')
            (left_pos, right_pos) = ([], [])
            for s in sorted(idx_rm):
                left_pos.append(input_left.find(s))
                right_pos.append(input_right.find(s))
            new_view = tensordot(*tmp_operands, axes=(tuple(left_pos), tuple(right_pos)))
            if tensor_result != results_index or handle_out:
                if handle_out:
                    kwargs['out'] = out
                new_view = c_einsum(tensor_result + '->' + results_index, new_view, **kwargs)
        else:
            if handle_out:
                kwargs['out'] = out
            new_view = c_einsum(einsum_str, *tmp_operands, **kwargs)
        operands.append(new_view)
        del tmp_operands, new_view
    if specified_out:
        return out
    else:
        return asanyarray(operands[0], order=output_order)