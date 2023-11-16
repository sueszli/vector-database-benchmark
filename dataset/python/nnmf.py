from scipy.sparse import dok_matrix, csc_matrix, rand
import numpy as np


def build_sparse_matrix(list_of_dicts, vector_length, orient='columns', verbose=False):
    """
    Function for building sparse matrix from list of dicts
    :param list_of_dicts: list of dictionaries representing sparse vectors
    :param vector_length: number of values in dense representation of sparse vector
    :param orient: build matrix by rows or columns - default is columns
    :return: sparse matrix
    """
    if orient == 'columns':
        columns = len(list_of_dicts)
        matrix = dok_matrix((vector_length, columns))
        for column, vector in enumerate(list_of_dicts):
            if verbose:
                print("Building matrix {:0.2%}".format(column / columns), end='\r')
            for term in vector.keys():
                matrix[int(term), column] = vector[term]
    elif orient == 'rows':
        rows = len(list_of_dicts)
        matrix = dok_matrix(shape=(rows, vector_length))
        for row, vector in enumerate(list_of_dicts):
            if verbose:
                print("Building matrix {:0.2%}".format(row / rows), end='\r')
            for term in vector.keys():
                matrix[row, term] = vector[term]
    else:
        raise ValueError('Orient must be either \'columns\' or \'rows\'')

    print("Matrix complete.                    ")
    return csc_matrix(matrix)


def cost(a, b):
    """
    Function takes two sparse matrices and
    returns total Euclidian distance between all vectors
    :param a: sparse matrix 1
    :param b: sparse matrix 2
    :return: Euclidian distance
    """
    diff = a - b
    diff = diff.multiply(diff)
    diff = diff.sum()
    return diff


def factorise(V, topics=10, iterations=50, init_density=0.01, convergence=None):
    """
    Factorise function computes Non-negative Matrix Factorisation of input data
    :param V: input data matrix (data instances (tweets) are columns
    :param topics: number of topics required in output
    :param iterations: maximum number of training iterations
    :param init_density: density of initialised weight matrices W and H (proportion or non-zero values)
    :return W: component feature matrix - component vectors found in columns of matrix
    :return H: matrix for reconstruction of original data from component features
    """

    terms = V.shape[0]
    instances = V.shape[1]

    # Initialize the weight and feature matrices with random values
    # W: terms x topics sized matrix
    W = rand(terms, topics, density=init_density, format='csc')
    # H: topics x instances sized matrix
    H = rand(topics, instances, density=init_density, format='csc')

    cost_history = []
    cache_cost = np.inf

    # Repeat iterative algorithm maximum 'iterations' number of times
    for i in range(iterations):
        print("Iteration: {}/{}       ".format(i + 1, iterations), end='\r')
        # E step
        # WH: terms x instances sized matrix
        WH = W * H

        # Calculate the current difference between factorisation and actual
        temp_cost = cost(V, WH)

        cost_history.append(temp_cost)

        # End if matrix perfectly factorised or reaches convergence criteria
        if temp_cost == 0:
            break
        if convergence is not None and cache_cost - temp_cost < convergence:
            print("Met convergence criteria of {} on iteration {}".format(convergence, i+1))
            break
        else:
            cache_cost = temp_cost

        # Update weights matrix
        # W_numerator: terms x topics matrix
        W_numerator = V * H.transpose()
        # W_denominator: terms x topics matrix
        W_denominator = W * H * H.transpose()
        W_denominator.data[:] = 1 / W_denominator.data

        # W: terms x topics matrix
        W = W.multiply(W_numerator).multiply(W_denominator)
        W = csc_matrix(W.multiply(1 / W.sum(axis=0)))

        # Update feature matrix
        # H_numerator: topics x instances matrix
        H_numerator = W.transpose() * V
        # H_denominator: topics x instances matrix
        H_denominator = W.transpose() * W * H
        H_denominator.data[:] = 1 / H_denominator.data

        # H: topics x instances matrix
        H = H.multiply(H_numerator).multiply(H_denominator)

    print('Factorisation successful.\n')
    print('Error profile: {}\n'.format(cost_history))
    return dok_matrix(W), H


def evaluate(W, term_dict, print_output=True):
    """
    Evaluate W matrix from nnmf,
    :param W: W matrix
    :param term_dict: id to term reference dictionary
    :return: list of topics containing terms and relative values
    """
    items = W.items()
    items = sorted(items, key=lambda x: x[1], reverse=True)
    topics = [[] for i in range(W.shape[1])]
    for index, value in items:
        term_value = (term_dict[str(index[0])], value)
        topics[index[1]].append(term_value)
    if print_output:
        for i, t in enumerate(topics):
            print("Topic {}: ".format(i+1))
            for term, value in t[:-1]:
                print(term + ",", end=' ')
            print('{}\n'.format(t[-1][0]))

    return topics
