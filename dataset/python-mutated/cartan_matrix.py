from .cartan_type import CartanType

def CartanMatrix(ct):
    if False:
        for i in range(10):
            print('nop')
    'Access the Cartan matrix of a specific Lie algebra\n\n    Examples\n    ========\n\n    >>> from sympy.liealgebras.cartan_matrix import CartanMatrix\n    >>> CartanMatrix("A2")\n    Matrix([\n    [ 2, -1],\n    [-1,  2]])\n\n    >>> CartanMatrix([\'C\', 3])\n    Matrix([\n    [ 2, -1,  0],\n    [-1,  2, -1],\n    [ 0, -2,  2]])\n\n    This method works by returning the Cartan matrix\n    which corresponds to Cartan type t.\n    '
    return CartanType(ct).cartan_matrix()