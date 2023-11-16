"""
This is the core of deep learning: (1) Take an input and desired output, (2) Search for their correlation
"""

def compute_error(b, m, coordinates):
    if False:
        for i in range(10):
            print('nop')
    '\n    m is the coefficient and b is the constant for prediction\n    The goal is to find a combination of m and b where the error is as small as possible\n    coordinates are the locations\n    '
    totalError = 0
    for i in range(0, len(coordinates)):
        x = coordinates[i][0]
        y = coordinates[i][1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(coordinates))
error = compute_error(1, 2, [[3, 6], [6, 9], [12, 18]])
print(error)