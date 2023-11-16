"""
Dynamic Programming
Implementation of matrix Chain Multiplication
Time Complexity: O(n^3)
Space Complexity: O(n^2)
"""
INF = float('inf')

def matrix_chain_order(array):
    if False:
        print('Hello World!')
    'Finds optimal order to multiply matrices\n\n    array -- int[]\n    '
    n = len(array)
    matrix = [[0 for x in range(n)] for x in range(n)]
    sol = [[0 for x in range(n)] for x in range(n)]
    for chain_length in range(2, n):
        for a in range(1, n - chain_length + 1):
            b = a + chain_length - 1
            matrix[a][b] = INF
            for c in range(a, b):
                cost = matrix[a][c] + matrix[c + 1][b] + array[a - 1] * array[c] * array[b]
                if cost < matrix[a][b]:
                    matrix[a][b] = cost
                    sol[a][b] = c
    return (matrix, sol)

def print_optimal_solution(optimal_solution, i, j):
    if False:
        print('Hello World!')
    'Print the solution\n\n    optimal_solution -- int[][]\n    i -- int[]\n    j -- int[]\n    '
    if i == j:
        print('A' + str(i), end=' ')
    else:
        print('(', end=' ')
        print_optimal_solution(optimal_solution, i, optimal_solution[i][j])
        print_optimal_solution(optimal_solution, optimal_solution[i][j] + 1, j)
        print(')', end=' ')

def main():
    if False:
        print('Hello World!')
    '\n    Testing for matrix_chain_ordering\n    '
    array = [30, 35, 15, 5, 10, 20, 25]
    length = len(array)
    (matrix, optimal_solution) = matrix_chain_order(array)
    print('No. of Operation required: ' + str(matrix[1][length - 1]))
    print_optimal_solution(optimal_solution, 1, length - 1)
if __name__ == '__main__':
    main()