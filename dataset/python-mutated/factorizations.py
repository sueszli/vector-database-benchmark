from sympy.matrices.expressions import MatrixExpr
from sympy.assumptions.ask import Q

class Factorization(MatrixExpr):
    arg = property(lambda self: self.args[0])
    shape = property(lambda self: self.arg.shape)

class LofLU(Factorization):

    @property
    def predicates(self):
        if False:
            while True:
                i = 10
        return (Q.lower_triangular,)

class UofLU(Factorization):

    @property
    def predicates(self):
        if False:
            i = 10
            return i + 15
        return (Q.upper_triangular,)

class LofCholesky(LofLU):
    pass

class UofCholesky(UofLU):
    pass

class QofQR(Factorization):

    @property
    def predicates(self):
        if False:
            i = 10
            return i + 15
        return (Q.orthogonal,)

class RofQR(Factorization):

    @property
    def predicates(self):
        if False:
            print('Hello World!')
        return (Q.upper_triangular,)

class EigenVectors(Factorization):

    @property
    def predicates(self):
        if False:
            while True:
                i = 10
        return (Q.orthogonal,)

class EigenValues(Factorization):

    @property
    def predicates(self):
        if False:
            return 10
        return (Q.diagonal,)

class UofSVD(Factorization):

    @property
    def predicates(self):
        if False:
            return 10
        return (Q.orthogonal,)

class SofSVD(Factorization):

    @property
    def predicates(self):
        if False:
            i = 10
            return i + 15
        return (Q.diagonal,)

class VofSVD(Factorization):

    @property
    def predicates(self):
        if False:
            print('Hello World!')
        return (Q.orthogonal,)

def lu(expr):
    if False:
        print('Hello World!')
    return (LofLU(expr), UofLU(expr))

def qr(expr):
    if False:
        while True:
            i = 10
    return (QofQR(expr), RofQR(expr))

def eig(expr):
    if False:
        return 10
    return (EigenValues(expr), EigenVectors(expr))

def svd(expr):
    if False:
        i = 10
        return i + 15
    return (UofSVD(expr), SofSVD(expr), VofSVD(expr))