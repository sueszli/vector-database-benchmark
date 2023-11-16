from scipy.sparse import dok_matrix, csc_matrix, rand
import numpy as np

def build_sparse_matrix(list_of_dicts, vector_length, orient='columns', verbose=False):
    if False:
        print('Hello World!')
    '\n    Function for building sparse matrix from list of dicts\n    :param list_of_dicts: list of dictionaries representing sparse vectors\n    :param vector_length: number of values in dense representation of sparse vector\n    :param orient: build matrix by rows or columns - default is columns\n    :return: sparse matrix\n    '
    if orient == 'columns':
        columns = len(list_of_dicts)
        matrix = dok_matrix((vector_length, columns))
        for (column, vector) in enumerate(list_of_dicts):
            if verbose:
                print('Building matrix {:0.2%}'.format(column / columns), end='\r')
            for term in vector.keys():
                matrix[int(term), column] = vector[term]
    elif orient == 'rows':
        rows = len(list_of_dicts)
        matrix = dok_matrix(shape=(rows, vector_length))
        for (row, vector) in enumerate(list_of_dicts):
            if verbose:
                print('Building matrix {:0.2%}'.format(row / rows), end='\r')
            for term in vector.keys():
                matrix[row, term] = vector[term]
    else:
        raise ValueError("Orient must be either 'columns' or 'rows'")
    print('Matrix complete.                    ')
    return csc_matrix(matrix)

def cost(a, b):
    if False:
        return 10
    '\n    Function takes two sparse matrices and\n    returns total Euclidian distance between all vectors\n    :param a: sparse matrix 1\n    :param b: sparse matrix 2\n    :return: Euclidian distance\n    '
    diff = a - b
    diff = diff.multiply(diff)
    diff = diff.sum()
    return diff

def factorise(V, topics=10, iterations=50, init_density=0.01, convergence=None):
    if False:
        i = 10
        return i + 15
    '\n    Factorise function computes Non-negative Matrix Factorisation of input data\n    :param V: input data matrix (data instances (tweets) are columns\n    :param topics: number of topics required in output\n    :param iterations: maximum number of training iterations\n    :param init_density: density of initialised weight matrices W and H (proportion or non-zero values)\n    :return W: component feature matrix - component vectors found in columns of matrix\n    :return H: matrix for reconstruction of original data from component features\n    '
    terms = V.shape[0]
    instances = V.shape[1]
    W = rand(terms, topics, density=init_density, format='csc')
    H = rand(topics, instances, density=init_density, format='csc')
    cost_history = []
    cache_cost = np.inf
    for i in range(iterations):
        print('Iteration: {}/{}       '.format(i + 1, iterations), end='\r')
        WH = W * H
        temp_cost = cost(V, WH)
        cost_history.append(temp_cost)
        if temp_cost == 0:
            break
        if convergence is not None and cache_cost - temp_cost < convergence:
            print('Met convergence criteria of {} on iteration {}'.format(convergence, i + 1))
            break
        else:
            cache_cost = temp_cost
        W_numerator = V * H.transpose()
        W_denominator = W * H * H.transpose()
        W_denominator.data[:] = 1 / W_denominator.data
        W = W.multiply(W_numerator).multiply(W_denominator)
        W = csc_matrix(W.multiply(1 / W.sum(axis=0)))
        H_numerator = W.transpose() * V
        H_denominator = W.transpose() * W * H
        H_denominator.data[:] = 1 / H_denominator.data
        H = H.multiply(H_numerator).multiply(H_denominator)
    print('Factorisation successful.\n')
    print('Error profile: {}\n'.format(cost_history))
    return (dok_matrix(W), H)

def evaluate(W, term_dict, print_output=True):
    if False:
        print('Hello World!')
    '\n    Evaluate W matrix from nnmf,\n    :param W: W matrix\n    :param term_dict: id to term reference dictionary\n    :return: list of topics containing terms and relative values\n    '
    items = W.items()
    items = sorted(items, key=lambda x: x[1], reverse=True)
    topics = [[] for i in range(W.shape[1])]
    for (index, value) in items:
        term_value = (term_dict[str(index[0])], value)
        topics[index[1]].append(term_value)
    if print_output:
        for (i, t) in enumerate(topics):
            print('Topic {}: '.format(i + 1))
            for (term, value) in t[:-1]:
                print(term + ',', end=' ')
            print('{}\n'.format(t[-1][0]))
    return topics