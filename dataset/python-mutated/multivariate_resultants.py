"""
This module contains functions for two multivariate resultants. These
are:

- Dixon's resultant.
- Macaulay's resultant.

Multivariate resultants are used to identify whether a multivariate
system has common roots. That is when the resultant is equal to zero.
"""
from math import prod
from sympy.core.mul import Mul
from sympy.matrices.dense import Matrix, diag
from sympy.polys.polytools import Poly, degree_list, rem
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import IndexedBase
from sympy.polys.monomials import itermonomials, monomial_deg
from sympy.polys.orderings import monomial_key
from sympy.polys.polytools import poly_from_expr, total_degree
from sympy.functions.combinatorial.factorials import binomial
from itertools import combinations_with_replacement
from sympy.utilities.exceptions import sympy_deprecation_warning

class DixonResultant:
    """
    A class for retrieving the Dixon's resultant of a multivariate
    system.

    Examples
    ========

    >>> from sympy import symbols

    >>> from sympy.polys.multivariate_resultants import DixonResultant
    >>> x, y = symbols('x, y')

    >>> p = x + y
    >>> q = x ** 2 + y ** 3
    >>> h = x ** 2 + y

    >>> dixon = DixonResultant(variables=[x, y], polynomials=[p, q, h])
    >>> poly = dixon.get_dixon_polynomial()
    >>> matrix = dixon.get_dixon_matrix(polynomial=poly)
    >>> matrix
    Matrix([
    [ 0,  0, -1,  0, -1],
    [ 0, -1,  0, -1,  0],
    [-1,  0,  1,  0,  0],
    [ 0, -1,  0,  0,  1],
    [-1,  0,  0,  1,  0]])
    >>> matrix.det()
    0

    See Also
    ========

    Notebook in examples: sympy/example/notebooks.

    References
    ==========

    .. [1] [Kapur1994]_
    .. [2] [Palancz08]_

    """

    def __init__(self, polynomials, variables):
        if False:
            i = 10
            return i + 15
        '\n        A class that takes two lists, a list of polynomials and list of\n        variables. Returns the Dixon matrix of the multivariate system.\n\n        Parameters\n        ----------\n        polynomials : list of polynomials\n            A  list of m n-degree polynomials\n        variables: list\n            A list of all n variables\n        '
        self.polynomials = polynomials
        self.variables = variables
        self.n = len(self.variables)
        self.m = len(self.polynomials)
        a = IndexedBase('alpha')
        self.dummy_variables = [a[i] for i in range(self.n)]
        self._max_degrees = [max((degree_list(poly)[i] for poly in self.polynomials)) for i in range(self.n)]

    @property
    def max_degrees(self):
        if False:
            for i in range(10):
                print('nop')
        sympy_deprecation_warning('\n            The max_degrees property of DixonResultant is deprecated.\n            ', deprecated_since_version='1.5', active_deprecations_target='deprecated-dixonresultant-properties')
        return self._max_degrees

    def get_dixon_polynomial(self):
        if False:
            while True:
                i = 10
        "\n        Returns\n        =======\n\n        dixon_polynomial: polynomial\n            Dixon's polynomial is calculated as:\n\n            delta = Delta(A) / ((x_1 - a_1) ... (x_n - a_n)) where,\n\n            A =  |p_1(x_1,... x_n), ..., p_n(x_1,... x_n)|\n                 |p_1(a_1,... x_n), ..., p_n(a_1,... x_n)|\n                 |...             , ...,              ...|\n                 |p_1(a_1,... a_n), ..., p_n(a_1,... a_n)|\n        "
        if self.m != self.n + 1:
            raise ValueError('Method invalid for given combination.')
        rows = [self.polynomials]
        temp = list(self.variables)
        for idx in range(self.n):
            temp[idx] = self.dummy_variables[idx]
            substitution = dict(zip(self.variables, temp))
            rows.append([f.subs(substitution) for f in self.polynomials])
        A = Matrix(rows)
        terms = zip(self.variables, self.dummy_variables)
        product_of_differences = Mul(*[a - b for (a, b) in terms])
        dixon_polynomial = (A.det() / product_of_differences).factor()
        return poly_from_expr(dixon_polynomial, self.dummy_variables)[0]

    def get_upper_degree(self):
        if False:
            i = 10
            return i + 15
        sympy_deprecation_warning('\n            The get_upper_degree() method of DixonResultant is deprecated. Use\n            get_max_degrees() instead.\n            ', deprecated_since_version='1.5', active_deprecations_target='deprecated-dixonresultant-properties')
        list_of_products = [self.variables[i] ** self._max_degrees[i] for i in range(self.n)]
        product = prod(list_of_products)
        product = Poly(product).monoms()
        return monomial_deg(*product)

    def get_max_degrees(self, polynomial):
        if False:
            return 10
        '\n        Returns a list of the maximum degree of each variable appearing\n        in the coefficients of the Dixon polynomial. The coefficients are\n        viewed as polys in $x_1, x_2, \\dots, x_n$.\n        '
        deg_lists = [degree_list(Poly(poly, self.variables)) for poly in polynomial.coeffs()]
        max_degrees = [max(degs) for degs in zip(*deg_lists)]
        return max_degrees

    def get_dixon_matrix(self, polynomial):
        if False:
            return 10
        '\n        Construct the Dixon matrix from the coefficients of polynomial\n        \\alpha. Each coefficient is viewed as a polynomial of x_1, ...,\n        x_n.\n        '
        max_degrees = self.get_max_degrees(polynomial)
        monomials = itermonomials(self.variables, max_degrees)
        monomials = sorted(monomials, reverse=True, key=monomial_key('lex', self.variables))
        dixon_matrix = Matrix([[Poly(c, *self.variables).coeff_monomial(m) for m in monomials] for c in polynomial.coeffs()])
        if dixon_matrix.shape[0] != dixon_matrix.shape[1]:
            keep = [column for column in range(dixon_matrix.shape[-1]) if any((element != 0 for element in dixon_matrix[:, column]))]
            dixon_matrix = dixon_matrix[:, keep]
        return dixon_matrix

    def KSY_precondition(self, matrix):
        if False:
            print('Hello World!')
        '\n        Test for the validity of the Kapur-Saxena-Yang precondition.\n\n        The precondition requires that the column corresponding to the\n        monomial 1 = x_1 ^ 0 * x_2 ^ 0 * ... * x_n ^ 0 is not a linear\n        combination of the remaining ones. In SymPy notation this is\n        the last column. For the precondition to hold the last non-zero\n        row of the rref matrix should be of the form [0, 0, ..., 1].\n        '
        if matrix.is_zero_matrix:
            return False
        (m, n) = matrix.shape
        matrix = simplify(matrix.rref()[0])
        rows = [i for i in range(m) if any((matrix[i, j] != 0 for j in range(n)))]
        matrix = matrix[rows, :]
        condition = Matrix([[0] * (n - 1) + [1]])
        if matrix[-1, :] == condition:
            return True
        else:
            return False

    def delete_zero_rows_and_columns(self, matrix):
        if False:
            i = 10
            return i + 15
        'Remove the zero rows and columns of the matrix.'
        rows = [i for i in range(matrix.rows) if not matrix.row(i).is_zero_matrix]
        cols = [j for j in range(matrix.cols) if not matrix.col(j).is_zero_matrix]
        return matrix[rows, cols]

    def product_leading_entries(self, matrix):
        if False:
            i = 10
            return i + 15
        'Calculate the product of the leading entries of the matrix.'
        res = 1
        for row in range(matrix.rows):
            for el in matrix.row(row):
                if el != 0:
                    res = res * el
                    break
        return res

    def get_KSY_Dixon_resultant(self, matrix):
        if False:
            i = 10
            return i + 15
        'Calculate the Kapur-Saxena-Yang approach to the Dixon Resultant.'
        matrix = self.delete_zero_rows_and_columns(matrix)
        (_, U, _) = matrix.LUdecomposition()
        matrix = self.delete_zero_rows_and_columns(simplify(U))
        return self.product_leading_entries(matrix)

class MacaulayResultant:
    """
    A class for calculating the Macaulay resultant. Note that the
    polynomials must be homogenized and their coefficients must be
    given as symbols.

    Examples
    ========

    >>> from sympy import symbols

    >>> from sympy.polys.multivariate_resultants import MacaulayResultant
    >>> x, y, z = symbols('x, y, z')

    >>> a_0, a_1, a_2 = symbols('a_0, a_1, a_2')
    >>> b_0, b_1, b_2 = symbols('b_0, b_1, b_2')
    >>> c_0, c_1, c_2,c_3, c_4 = symbols('c_0, c_1, c_2, c_3, c_4')

    >>> f = a_0 * y -  a_1 * x + a_2 * z
    >>> g = b_1 * x ** 2 + b_0 * y ** 2 - b_2 * z ** 2
    >>> h = c_0 * y * z ** 2 - c_1 * x ** 3 + c_2 * x ** 2 * z - c_3 * x * z ** 2 + c_4 * z ** 3

    >>> mac = MacaulayResultant(polynomials=[f, g, h], variables=[x, y, z])
    >>> mac.monomial_set
    [x**4, x**3*y, x**3*z, x**2*y**2, x**2*y*z, x**2*z**2, x*y**3,
    x*y**2*z, x*y*z**2, x*z**3, y**4, y**3*z, y**2*z**2, y*z**3, z**4]
    >>> matrix = mac.get_matrix()
    >>> submatrix = mac.get_submatrix(matrix)
    >>> submatrix
    Matrix([
    [-a_1,  a_0,  a_2,    0],
    [   0, -a_1,    0,    0],
    [   0,    0, -a_1,    0],
    [   0,    0,    0, -a_1]])

    See Also
    ========

    Notebook in examples: sympy/example/notebooks.

    References
    ==========

    .. [1] [Bruce97]_
    .. [2] [Stiller96]_

    """

    def __init__(self, polynomials, variables):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ==========\n\n        variables: list\n            A list of all n variables\n        polynomials : list of SymPy polynomials\n            A  list of m n-degree polynomials\n        '
        self.polynomials = polynomials
        self.variables = variables
        self.n = len(variables)
        self.degrees = [total_degree(poly, *self.variables) for poly in self.polynomials]
        self.degree_m = self._get_degree_m()
        self.monomials_size = self.get_size()
        self.monomial_set = self.get_monomials_of_certain_degree(self.degree_m)

    def _get_degree_m(self):
        if False:
            while True:
                i = 10
        '\n        Returns\n        =======\n\n        degree_m: int\n            The degree_m is calculated as  1 + \\sum_1 ^ n (d_i - 1),\n            where d_i is the degree of the i polynomial\n        '
        return 1 + sum((d - 1 for d in self.degrees))

    def get_size(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns\n        =======\n\n        size: int\n            The size of set T. Set T is the set of all possible\n            monomials of the n variables for degree equal to the\n            degree_m\n        '
        return binomial(self.degree_m + self.n - 1, self.n - 1)

    def get_monomials_of_certain_degree(self, degree):
        if False:
            print('Hello World!')
        '\n        Returns\n        =======\n\n        monomials: list\n            A list of monomials of a certain degree.\n        '
        monomials = [Mul(*monomial) for monomial in combinations_with_replacement(self.variables, degree)]
        return sorted(monomials, reverse=True, key=monomial_key('lex', self.variables))

    def get_row_coefficients(self):
        if False:
            print('Hello World!')
        "\n        Returns\n        =======\n\n        row_coefficients: list\n            The row coefficients of Macaulay's matrix\n        "
        row_coefficients = []
        divisible = []
        for i in range(self.n):
            if i == 0:
                degree = self.degree_m - self.degrees[i]
                monomial = self.get_monomials_of_certain_degree(degree)
                row_coefficients.append(monomial)
            else:
                divisible.append(self.variables[i - 1] ** self.degrees[i - 1])
                degree = self.degree_m - self.degrees[i]
                poss_rows = self.get_monomials_of_certain_degree(degree)
                for div in divisible:
                    for p in poss_rows:
                        if rem(p, div) == 0:
                            poss_rows = [item for item in poss_rows if item != p]
                row_coefficients.append(poss_rows)
        return row_coefficients

    def get_matrix(self):
        if False:
            return 10
        '\n        Returns\n        =======\n\n        macaulay_matrix: Matrix\n            The Macaulay numerator matrix\n        '
        rows = []
        row_coefficients = self.get_row_coefficients()
        for i in range(self.n):
            for multiplier in row_coefficients[i]:
                coefficients = []
                poly = Poly(self.polynomials[i] * multiplier, *self.variables)
                for mono in self.monomial_set:
                    coefficients.append(poly.coeff_monomial(mono))
                rows.append(coefficients)
        macaulay_matrix = Matrix(rows)
        return macaulay_matrix

    def get_reduced_nonreduced(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns\n        =======\n\n        reduced: list\n            A list of the reduced monomials\n        non_reduced: list\n            A list of the monomials that are not reduced\n\n        Definition\n        ==========\n\n        A polynomial is said to be reduced in x_i, if its degree (the\n        maximum degree of its monomials) in x_i is less than d_i. A\n        polynomial that is reduced in all variables but one is said\n        simply to be reduced.\n        '
        divisible = []
        for m in self.monomial_set:
            temp = []
            for (i, v) in enumerate(self.variables):
                temp.append(bool(total_degree(m, v) >= self.degrees[i]))
            divisible.append(temp)
        reduced = [i for (i, r) in enumerate(divisible) if sum(r) < self.n - 1]
        non_reduced = [i for (i, r) in enumerate(divisible) if sum(r) >= self.n - 1]
        return (reduced, non_reduced)

    def get_submatrix(self, matrix):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns\n        =======\n\n        macaulay_submatrix: Matrix\n            The Macaulay denominator matrix. Columns that are non reduced are kept.\n            The row which contains one of the a_{i}s is dropped. a_{i}s\n            are the coefficients of x_i ^ {d_i}.\n        '
        (reduced, non_reduced) = self.get_reduced_nonreduced()
        if reduced == []:
            return diag([1])
        reduction_set = [v ** self.degrees[i] for (i, v) in enumerate(self.variables)]
        ais = [self.polynomials[i].coeff(reduction_set[i]) for i in range(self.n)]
        reduced_matrix = matrix[:, reduced]
        keep = []
        for row in range(reduced_matrix.rows):
            check = [ai in reduced_matrix[row, :] for ai in ais]
            if True not in check:
                keep.append(row)
        return matrix[keep, non_reduced]