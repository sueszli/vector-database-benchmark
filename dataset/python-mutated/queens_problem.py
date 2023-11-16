"""
Queens Problem

You have an N by N board. Write a function that, given N, returns the number of possible arrangements
of the board where N queens can be placed on the board without threatening each other,
i.e. no two queens share the same row, column, or diagonal.

=========================================
Backtracking solution.
    Time Complexity:    O(N!) (but I think it's much faster!)
    Space Complexity:   O(N)
* There are much faster solutions, like O(N^2)
"""

def place_n_queens(n):
    if False:
        print('Hello World!')
    columns = [False for i in range(n)]
    order = []
    return backtracking(columns, order)

def backtracking(columns, order):
    if False:
        return 10
    n = len(columns)
    if len(order) == n:
        return 1
    total = 0
    for i in range(n):
        if not columns[i] and check_diagonals(order, i):
            order.append(i)
            columns[i] = True
            total += backtracking(columns, order)
            columns[i] = False
            del order[-1]
    return total

def check_diagonals(order, pos):
    if False:
        for i in range(10):
            print('nop')
    current_row = len(order)
    for i in range(current_row):
        if i - order[i] == current_row - pos:
            return False
        if i + order[i] == current_row + pos:
            return False
    return True
print(place_n_queens(1))
print(place_n_queens(2))
print(place_n_queens(3))
print(place_n_queens(4))
print(place_n_queens(5))
print(place_n_queens(6))
print(place_n_queens(7))
print(place_n_queens(8))
print(place_n_queens(9))
print(place_n_queens(10))