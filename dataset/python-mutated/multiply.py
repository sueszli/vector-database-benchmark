"""
This algorithm takes two compatible two dimensional matrix
and return their product
Space complexity: O(n^2)
Possible edge case: the number of columns of multiplicand not consistent with
the number of rows of multiplier, will raise exception
"""

def multiply(multiplicand: list, multiplier: list) -> list:
    if False:
        for i in range(10):
            print('nop')
    '\n    :type A: List[List[int]]\n    :type B: List[List[int]]\n    :rtype: List[List[int]]\n    '
    (multiplicand_row, multiplicand_col) = (len(multiplicand), len(multiplicand[0]))
    (multiplier_row, multiplier_col) = (len(multiplier), len(multiplier[0]))
    if multiplicand_col != multiplier_row:
        raise Exception('Multiplicand matrix not compatible with Multiplier matrix.')
    result = [[0] * multiplier_col for i in range(multiplicand_row)]
    for i in range(multiplicand_row):
        for j in range(multiplier_col):
            for k in range(len(multiplier)):
                result[i][j] += multiplicand[i][k] * multiplier[k][j]
    return result