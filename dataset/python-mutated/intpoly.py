"""
Module to implement integration of uni/bivariate polynomials over
2D Polytopes and uni/bi/trivariate polynomials over 3D Polytopes.

Uses evaluation techniques as described in Chin et al. (2015) [1].


References
===========

.. [1] Chin, Eric B., Jean B. Lasserre, and N. Sukumar. "Numerical integration
of homogeneous functions on convex and nonconvex polygons and polyhedra."
Computational Mechanics 56.6 (2015): 967-981

PDF link : http://dilbert.engr.ucdavis.edu/~suku/quadrature/cls-integration.pdf
"""
from functools import cmp_to_key
from sympy.abc import x, y, z
from sympy.core import S, diff, Expr, Symbol
from sympy.core.sympify import _sympify
from sympy.geometry import Segment2D, Polygon, Point, Point2D
from sympy.polys.polytools import LC, gcd_list, degree_list, Poly
from sympy.simplify.simplify import nsimplify

def polytope_integrate(poly, expr=None, *, clockwise=False, max_degree=None):
    if False:
        print('Hello World!')
    'Integrates polynomials over 2/3-Polytopes.\n\n    Explanation\n    ===========\n\n    This function accepts the polytope in ``poly`` and the function in ``expr``\n    (uni/bi/trivariate polynomials are implemented) and returns\n    the exact integral of ``expr`` over ``poly``.\n\n    Parameters\n    ==========\n\n    poly : The input Polygon.\n\n    expr : The input polynomial.\n\n    clockwise : Binary value to sort input points of 2-Polytope clockwise.(Optional)\n\n    max_degree : The maximum degree of any monomial of the input polynomial.(Optional)\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y\n    >>> from sympy import Point, Polygon\n    >>> from sympy.integrals.intpoly import polytope_integrate\n    >>> polygon = Polygon(Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0))\n    >>> polys = [1, x, y, x*y, x**2*y, x*y**2]\n    >>> expr = x*y\n    >>> polytope_integrate(polygon, expr)\n    1/4\n    >>> polytope_integrate(polygon, polys, max_degree=3)\n    {1: 1, x: 1/2, y: 1/2, x*y: 1/4, x*y**2: 1/6, x**2*y: 1/6}\n    '
    if clockwise:
        if isinstance(poly, Polygon):
            poly = Polygon(*point_sort(poly.vertices), evaluate=False)
        else:
            raise TypeError('clockwise=True works for only 2-PolytopeV-representation input')
    if isinstance(poly, Polygon):
        hp_params = hyperplane_parameters(poly)
        facets = poly.sides
    elif len(poly[0]) == 2:
        plen = len(poly)
        if len(poly[0][0]) == 2:
            intersections = [intersection(poly[(i - 1) % plen], poly[i], 'plane2D') for i in range(0, plen)]
            hp_params = poly
            lints = len(intersections)
            facets = [Segment2D(intersections[i], intersections[(i + 1) % lints]) for i in range(lints)]
        else:
            raise NotImplementedError('Integration for H-representation 3Dcase not implemented yet.')
    else:
        vertices = poly[0]
        facets = poly[1:]
        hp_params = hyperplane_parameters(facets, vertices)
        if max_degree is None:
            if expr is None:
                raise TypeError('Input expression must be a valid SymPy expression')
            return main_integrate3d(expr, facets, vertices, hp_params)
    if max_degree is not None:
        result = {}
        if expr is not None:
            f_expr = []
            for e in expr:
                _ = decompose(e)
                if len(_) == 1 and (not _.popitem()[0]):
                    f_expr.append(e)
                elif Poly(e).total_degree() <= max_degree:
                    f_expr.append(e)
            expr = f_expr
        if not isinstance(expr, list) and expr is not None:
            raise TypeError('Input polynomials must be list of expressions')
        if len(hp_params[0][0]) == 3:
            result_dict = main_integrate3d(0, facets, vertices, hp_params, max_degree)
        else:
            result_dict = main_integrate(0, facets, hp_params, max_degree)
        if expr is None:
            return result_dict
        for poly in expr:
            poly = _sympify(poly)
            if poly not in result:
                if poly.is_zero:
                    result[S.Zero] = S.Zero
                    continue
                integral_value = S.Zero
                monoms = decompose(poly, separate=True)
                for monom in monoms:
                    monom = nsimplify(monom)
                    (coeff, m) = strip(monom)
                    integral_value += result_dict[m] * coeff
                result[poly] = integral_value
        return result
    if expr is None:
        raise TypeError('Input expression must be a valid SymPy expression')
    return main_integrate(expr, facets, hp_params)

def strip(monom):
    if False:
        i = 10
        return i + 15
    if monom.is_zero:
        return (S.Zero, S.Zero)
    elif monom.is_number:
        return (monom, S.One)
    else:
        coeff = LC(monom)
        return (coeff, monom / coeff)

def _polynomial_integrate(polynomials, facets, hp_params):
    if False:
        return 10
    dims = (x, y)
    dim_length = len(dims)
    integral_value = S.Zero
    for deg in polynomials:
        poly_contribute = S.Zero
        facet_count = 0
        for hp in hp_params:
            value_over_boundary = integration_reduction(facets, facet_count, hp[0], hp[1], polynomials[deg], dims, deg)
            poly_contribute += value_over_boundary * (hp[1] / norm(hp[0]))
            facet_count += 1
        poly_contribute /= dim_length + deg
        integral_value += poly_contribute
    return integral_value

def main_integrate3d(expr, facets, vertices, hp_params, max_degree=None):
    if False:
        i = 10
        return i + 15
    "Function to translate the problem of integrating uni/bi/tri-variate\n    polynomials over a 3-Polytope to integrating over its faces.\n    This is done using Generalized Stokes' Theorem and Euler's Theorem.\n\n    Parameters\n    ==========\n\n    expr :\n        The input polynomial.\n    facets :\n        Faces of the 3-Polytope(expressed as indices of `vertices`).\n    vertices :\n        Vertices that constitute the Polytope.\n    hp_params :\n        Hyperplane Parameters of the facets.\n    max_degree : optional\n        Max degree of constituent monomial in given list of polynomial.\n\n    Examples\n    ========\n\n    >>> from sympy.integrals.intpoly import main_integrate3d,     hyperplane_parameters\n    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),                (5, 0, 5), (5, 5, 0), (5, 5, 5)],                [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],                [3, 1, 0, 2], [0, 4, 6, 2]]\n    >>> vertices = cube[0]\n    >>> faces = cube[1:]\n    >>> hp_params = hyperplane_parameters(faces, vertices)\n    >>> main_integrate3d(1, faces, vertices, hp_params)\n    -125\n    "
    result = {}
    dims = (x, y, z)
    dim_length = len(dims)
    if max_degree:
        grad_terms = gradient_terms(max_degree, 3)
        flat_list = [term for z_terms in grad_terms for x_term in z_terms for term in x_term]
        for term in flat_list:
            result[term[0]] = 0
        for (facet_count, hp) in enumerate(hp_params):
            (a, b) = (hp[0], hp[1])
            x0 = vertices[facets[facet_count][0]]
            for (i, monom) in enumerate(flat_list):
                (expr, x_d, y_d, z_d, z_index, y_index, x_index, _) = monom
                degree = x_d + y_d + z_d
                if b.is_zero:
                    value_over_face = S.Zero
                else:
                    value_over_face = integration_reduction_dynamic(facets, facet_count, a, b, expr, degree, dims, x_index, y_index, z_index, x0, grad_terms, i, vertices, hp)
                monom[7] = value_over_face
                result[expr] += value_over_face * (b / norm(a)) / (dim_length + x_d + y_d + z_d)
        return result
    else:
        integral_value = S.Zero
        polynomials = decompose(expr)
        for deg in polynomials:
            poly_contribute = S.Zero
            facet_count = 0
            for (i, facet) in enumerate(facets):
                hp = hp_params[i]
                if hp[1].is_zero:
                    continue
                pi = polygon_integrate(facet, hp, i, facets, vertices, expr, deg)
                poly_contribute += pi * (hp[1] / norm(tuple(hp[0])))
                facet_count += 1
            poly_contribute /= dim_length + deg
            integral_value += poly_contribute
    return integral_value

def main_integrate(expr, facets, hp_params, max_degree=None):
    if False:
        for i in range(10):
            print('nop')
    "Function to translate the problem of integrating univariate/bivariate\n    polynomials over a 2-Polytope to integrating over its boundary facets.\n    This is done using Generalized Stokes's Theorem and Euler's Theorem.\n\n    Parameters\n    ==========\n\n    expr :\n        The input polynomial.\n    facets :\n        Facets(Line Segments) of the 2-Polytope.\n    hp_params :\n        Hyperplane Parameters of the facets.\n    max_degree : optional\n        The maximum degree of any monomial of the input polynomial.\n\n    >>> from sympy.abc import x, y\n    >>> from sympy.integrals.intpoly import main_integrate,    hyperplane_parameters\n    >>> from sympy import Point, Polygon\n    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))\n    >>> facets = triangle.sides\n    >>> hp_params = hyperplane_parameters(triangle)\n    >>> main_integrate(x**2 + y**2, facets, hp_params)\n    325/6\n    "
    dims = (x, y)
    dim_length = len(dims)
    result = {}
    if max_degree:
        grad_terms = [[0, 0, 0, 0]] + gradient_terms(max_degree)
        for (facet_count, hp) in enumerate(hp_params):
            (a, b) = (hp[0], hp[1])
            x0 = facets[facet_count].points[0]
            for (i, monom) in enumerate(grad_terms):
                (m, x_d, y_d, _) = monom
                value = result.get(m, None)
                degree = S.Zero
                if b.is_zero:
                    value_over_boundary = S.Zero
                else:
                    degree = x_d + y_d
                    value_over_boundary = integration_reduction_dynamic(facets, facet_count, a, b, m, degree, dims, x_d, y_d, max_degree, x0, grad_terms, i)
                monom[3] = value_over_boundary
                if value is not None:
                    result[m] += value_over_boundary * (b / norm(a)) / (dim_length + degree)
                else:
                    result[m] = value_over_boundary * (b / norm(a)) / (dim_length + degree)
        return result
    elif not isinstance(expr, list):
        polynomials = decompose(expr)
        return _polynomial_integrate(polynomials, facets, hp_params)
    else:
        return {e: _polynomial_integrate(decompose(e), facets, hp_params) for e in expr}

def polygon_integrate(facet, hp_param, index, facets, vertices, expr, degree):
    if False:
        for i in range(10):
            print('nop')
    'Helper function to integrate the input uni/bi/trivariate polynomial\n    over a certain face of the 3-Polytope.\n\n    Parameters\n    ==========\n\n    facet :\n        Particular face of the 3-Polytope over which ``expr`` is integrated.\n    index :\n        The index of ``facet`` in ``facets``.\n    facets :\n        Faces of the 3-Polytope(expressed as indices of `vertices`).\n    vertices :\n        Vertices that constitute the facet.\n    expr :\n        The input polynomial.\n    degree :\n        Degree of ``expr``.\n\n    Examples\n    ========\n\n    >>> from sympy.integrals.intpoly import polygon_integrate\n    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),                 (5, 0, 5), (5, 5, 0), (5, 5, 5)],                 [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],                 [3, 1, 0, 2], [0, 4, 6, 2]]\n    >>> facet = cube[1]\n    >>> facets = cube[1:]\n    >>> vertices = cube[0]\n    >>> polygon_integrate(facet, [(0, 1, 0), 5], 0, facets, vertices, 1, 0)\n    -25\n    '
    expr = S(expr)
    if expr.is_zero:
        return S.Zero
    result = S.Zero
    x0 = vertices[facet[0]]
    facet_len = len(facet)
    for (i, fac) in enumerate(facet):
        side = (vertices[fac], vertices[facet[(i + 1) % facet_len]])
        result += distance_to_side(x0, side, hp_param[0]) * lineseg_integrate(facet, i, side, expr, degree)
    if not expr.is_number:
        expr = diff(expr, x) * x0[0] + diff(expr, y) * x0[1] + diff(expr, z) * x0[2]
        result += polygon_integrate(facet, hp_param, index, facets, vertices, expr, degree - 1)
    result /= degree + 2
    return result

def distance_to_side(point, line_seg, A):
    if False:
        for i in range(10):
            print('nop')
    'Helper function to compute the signed distance between given 3D point\n    and a line segment.\n\n    Parameters\n    ==========\n\n    point : 3D Point\n    line_seg : Line Segment\n\n    Examples\n    ========\n\n    >>> from sympy.integrals.intpoly import distance_to_side\n    >>> point = (0, 0, 0)\n    >>> distance_to_side(point, [(0, 0, 1), (0, 1, 0)], (1, 0, 0))\n    -sqrt(2)/2\n    '
    (x1, x2) = line_seg
    rev_normal = [-1 * S(i) / norm(A) for i in A]
    vector = [x2[i] - x1[i] for i in range(0, 3)]
    vector = [vector[i] / norm(vector) for i in range(0, 3)]
    n_side = cross_product((0, 0, 0), rev_normal, vector)
    vectorx0 = [line_seg[0][i] - point[i] for i in range(0, 3)]
    dot_product = sum([vectorx0[i] * n_side[i] for i in range(0, 3)])
    return dot_product

def lineseg_integrate(polygon, index, line_seg, expr, degree):
    if False:
        i = 10
        return i + 15
    'Helper function to compute the line integral of ``expr`` over ``line_seg``.\n\n    Parameters\n    ===========\n\n    polygon :\n        Face of a 3-Polytope.\n    index :\n        Index of line_seg in polygon.\n    line_seg :\n        Line Segment.\n\n    Examples\n    ========\n\n    >>> from sympy.integrals.intpoly import lineseg_integrate\n    >>> polygon = [(0, 5, 0), (5, 5, 0), (5, 5, 5), (0, 5, 5)]\n    >>> line_seg = [(0, 5, 0), (5, 5, 0)]\n    >>> lineseg_integrate(polygon, 0, line_seg, 1, 0)\n    5\n    '
    expr = _sympify(expr)
    if expr.is_zero:
        return S.Zero
    result = S.Zero
    x0 = line_seg[0]
    distance = norm(tuple([line_seg[1][i] - line_seg[0][i] for i in range(3)]))
    if isinstance(expr, Expr):
        expr_dict = {x: line_seg[1][0], y: line_seg[1][1], z: line_seg[1][2]}
        result += distance * expr.subs(expr_dict)
    else:
        result += distance * expr
    expr = diff(expr, x) * x0[0] + diff(expr, y) * x0[1] + diff(expr, z) * x0[2]
    result += lineseg_integrate(polygon, index, line_seg, expr, degree - 1)
    result /= degree + 1
    return result

def integration_reduction(facets, index, a, b, expr, dims, degree):
    if False:
        i = 10
        return i + 15
    'Helper method for main_integrate. Returns the value of the input\n    expression evaluated over the polytope facet referenced by a given index.\n\n    Parameters\n    ===========\n\n    facets :\n        List of facets of the polytope.\n    index :\n        Index referencing the facet to integrate the expression over.\n    a :\n        Hyperplane parameter denoting direction.\n    b :\n        Hyperplane parameter denoting distance.\n    expr :\n        The expression to integrate over the facet.\n    dims :\n        List of symbols denoting axes.\n    degree :\n        Degree of the homogeneous polynomial.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y\n    >>> from sympy.integrals.intpoly import integration_reduction,    hyperplane_parameters\n    >>> from sympy import Point, Polygon\n    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))\n    >>> facets = triangle.sides\n    >>> a, b = hyperplane_parameters(triangle)[0]\n    >>> integration_reduction(facets, 0, a, b, 1, (x, y), 0)\n    5\n    '
    expr = _sympify(expr)
    if expr.is_zero:
        return expr
    value = S.Zero
    x0 = facets[index].points[0]
    m = len(facets)
    gens = (x, y)
    inner_product = diff(expr, gens[0]) * x0[0] + diff(expr, gens[1]) * x0[1]
    if inner_product != 0:
        value += integration_reduction(facets, index, a, b, inner_product, dims, degree - 1)
    value += left_integral2D(m, index, facets, x0, expr, gens)
    return value / (len(dims) + degree - 1)

def left_integral2D(m, index, facets, x0, expr, gens):
    if False:
        for i in range(10):
            print('nop')
    'Computes the left integral of Eq 10 in Chin et al.\n    For the 2D case, the integral is just an evaluation of the polynomial\n    at the intersection of two facets which is multiplied by the distance\n    between the first point of facet and that intersection.\n\n    Parameters\n    ==========\n\n    m :\n        No. of hyperplanes.\n    index :\n        Index of facet to find intersections with.\n    facets :\n        List of facets(Line Segments in 2D case).\n    x0 :\n        First point on facet referenced by index.\n    expr :\n        Input polynomial\n    gens :\n        Generators which generate the polynomial\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y\n    >>> from sympy.integrals.intpoly import left_integral2D\n    >>> from sympy import Point, Polygon\n    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))\n    >>> facets = triangle.sides\n    >>> left_integral2D(3, 0, facets, facets[0].points[0], 1, (x, y))\n    5\n    '
    value = S.Zero
    for j in range(m):
        intersect = ()
        if j in ((index - 1) % m, (index + 1) % m):
            intersect = intersection(facets[index], facets[j], 'segment2D')
        if intersect:
            distance_origin = norm(tuple(map(lambda x, y: x - y, intersect, x0)))
            if is_vertex(intersect):
                if isinstance(expr, Expr):
                    if len(gens) == 3:
                        expr_dict = {gens[0]: intersect[0], gens[1]: intersect[1], gens[2]: intersect[2]}
                    else:
                        expr_dict = {gens[0]: intersect[0], gens[1]: intersect[1]}
                    value += distance_origin * expr.subs(expr_dict)
                else:
                    value += distance_origin * expr
    return value

def integration_reduction_dynamic(facets, index, a, b, expr, degree, dims, x_index, y_index, max_index, x0, monomial_values, monom_index, vertices=None, hp_param=None):
    if False:
        for i in range(10):
            print('nop')
    "The same integration_reduction function which uses a dynamic\n    programming approach to compute terms by using the values of the integral\n    of previously computed terms.\n\n    Parameters\n    ==========\n\n    facets :\n        Facets of the Polytope.\n    index :\n        Index of facet to find intersections with.(Used in left_integral()).\n    a, b :\n        Hyperplane parameters.\n    expr :\n        Input monomial.\n    degree :\n        Total degree of ``expr``.\n    dims :\n        Tuple denoting axes variables.\n    x_index :\n        Exponent of 'x' in ``expr``.\n    y_index :\n        Exponent of 'y' in ``expr``.\n    max_index :\n        Maximum exponent of any monomial in ``monomial_values``.\n    x0 :\n        First point on ``facets[index]``.\n    monomial_values :\n        List of monomial values constituting the polynomial.\n    monom_index :\n        Index of monomial whose integration is being found.\n    vertices : optional\n        Coordinates of vertices constituting the 3-Polytope.\n    hp_param : optional\n        Hyperplane Parameter of the face of the facets[index].\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y\n    >>> from sympy.integrals.intpoly import (integration_reduction_dynamic,             hyperplane_parameters)\n    >>> from sympy import Point, Polygon\n    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))\n    >>> facets = triangle.sides\n    >>> a, b = hyperplane_parameters(triangle)[0]\n    >>> x0 = facets[0].points[0]\n    >>> monomial_values = [[0, 0, 0, 0], [1, 0, 0, 5],                           [y, 0, 1, 15], [x, 1, 0, None]]\n    >>> integration_reduction_dynamic(facets, 0, a, b, x, 1, (x, y), 1, 0, 1,                                      x0, monomial_values, 3)\n    25/2\n    "
    value = S.Zero
    m = len(facets)
    if expr == S.Zero:
        return expr
    if len(dims) == 2:
        if not expr.is_number:
            (_, x_degree, y_degree, _) = monomial_values[monom_index]
            x_index = monom_index - max_index + x_index - 2 if x_degree > 0 else 0
            y_index = monom_index - 1 if y_degree > 0 else 0
            (x_value, y_value) = (monomial_values[x_index][3], monomial_values[y_index][3])
            value += x_degree * x_value * x0[0] + y_degree * y_value * x0[1]
        value += left_integral2D(m, index, facets, x0, expr, dims)
    else:
        z_index = max_index
        if not expr.is_number:
            (x_degree, y_degree, z_degree) = (y_index, z_index - x_index - y_index, x_index)
            x_value = monomial_values[z_index - 1][y_index - 1][x_index][7] if x_degree > 0 else 0
            y_value = monomial_values[z_index - 1][y_index][x_index][7] if y_degree > 0 else 0
            z_value = monomial_values[z_index - 1][y_index][x_index - 1][7] if z_degree > 0 else 0
            value += x_degree * x_value * x0[0] + y_degree * y_value * x0[1] + z_degree * z_value * x0[2]
        value += left_integral3D(facets, index, expr, vertices, hp_param, degree)
    return value / (len(dims) + degree - 1)

def left_integral3D(facets, index, expr, vertices, hp_param, degree):
    if False:
        print('Hello World!')
    'Computes the left integral of Eq 10 in Chin et al.\n\n    Explanation\n    ===========\n\n    For the 3D case, this is the sum of the integral values over constituting\n    line segments of the face (which is accessed by facets[index]) multiplied\n    by the distance between the first point of facet and that line segment.\n\n    Parameters\n    ==========\n\n    facets :\n        List of faces of the 3-Polytope.\n    index :\n        Index of face over which integral is to be calculated.\n    expr :\n        Input polynomial.\n    vertices :\n        List of vertices that constitute the 3-Polytope.\n    hp_param :\n        The hyperplane parameters of the face.\n    degree :\n        Degree of the ``expr``.\n\n    Examples\n    ========\n\n    >>> from sympy.integrals.intpoly import left_integral3D\n    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),                 (5, 0, 5), (5, 5, 0), (5, 5, 5)],                 [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],                 [3, 1, 0, 2], [0, 4, 6, 2]]\n    >>> facets = cube[1:]\n    >>> vertices = cube[0]\n    >>> left_integral3D(facets, 3, 1, vertices, ([0, -1, 0], -5), 0)\n    -50\n    '
    value = S.Zero
    facet = facets[index]
    x0 = vertices[facet[0]]
    facet_len = len(facet)
    for (i, fac) in enumerate(facet):
        side = (vertices[fac], vertices[facet[(i + 1) % facet_len]])
        value += distance_to_side(x0, side, hp_param[0]) * lineseg_integrate(facet, i, side, expr, degree)
    return value

def gradient_terms(binomial_power=0, no_of_gens=2):
    if False:
        return 10
    'Returns a list of all the possible monomials between\n    0 and y**binomial_power for 2D case and z**binomial_power\n    for 3D case.\n\n    Parameters\n    ==========\n\n    binomial_power :\n        Power upto which terms are generated.\n    no_of_gens :\n        Denotes whether terms are being generated for 2D or 3D case.\n\n    Examples\n    ========\n\n    >>> from sympy.integrals.intpoly import gradient_terms\n    >>> gradient_terms(2)\n    [[1, 0, 0, 0], [y, 0, 1, 0], [y**2, 0, 2, 0], [x, 1, 0, 0],\n    [x*y, 1, 1, 0], [x**2, 2, 0, 0]]\n    >>> gradient_terms(2, 3)\n    [[[[1, 0, 0, 0, 0, 0, 0, 0]]], [[[y, 0, 1, 0, 1, 0, 0, 0],\n    [z, 0, 0, 1, 1, 0, 1, 0]], [[x, 1, 0, 0, 1, 1, 0, 0]]],\n    [[[y**2, 0, 2, 0, 2, 0, 0, 0], [y*z, 0, 1, 1, 2, 0, 1, 0],\n    [z**2, 0, 0, 2, 2, 0, 2, 0]], [[x*y, 1, 1, 0, 2, 1, 0, 0],\n    [x*z, 1, 0, 1, 2, 1, 1, 0]], [[x**2, 2, 0, 0, 2, 2, 0, 0]]]]\n    '
    if no_of_gens == 2:
        count = 0
        terms = [None] * int((binomial_power ** 2 + 3 * binomial_power + 2) / 2)
        for x_count in range(0, binomial_power + 1):
            for y_count in range(0, binomial_power - x_count + 1):
                terms[count] = [x ** x_count * y ** y_count, x_count, y_count, 0]
                count += 1
    else:
        terms = [[[[x ** x_count * y ** y_count * z ** (z_count - y_count - x_count), x_count, y_count, z_count - y_count - x_count, z_count, x_count, z_count - y_count - x_count, 0] for y_count in range(z_count - x_count, -1, -1)] for x_count in range(0, z_count + 1)] for z_count in range(0, binomial_power + 1)]
    return terms

def hyperplane_parameters(poly, vertices=None):
    if False:
        i = 10
        return i + 15
    'A helper function to return the hyperplane parameters\n    of which the facets of the polytope are a part of.\n\n    Parameters\n    ==========\n\n    poly :\n        The input 2/3-Polytope.\n    vertices :\n        Vertex indices of 3-Polytope.\n\n    Examples\n    ========\n\n    >>> from sympy import Point, Polygon\n    >>> from sympy.integrals.intpoly import hyperplane_parameters\n    >>> hyperplane_parameters(Polygon(Point(0, 3), Point(5, 3), Point(1, 1)))\n    [((0, 1), 3), ((1, -2), -1), ((-2, -1), -3)]\n    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),                (5, 0, 5), (5, 5, 0), (5, 5, 5)],                [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],                [3, 1, 0, 2], [0, 4, 6, 2]]\n    >>> hyperplane_parameters(cube[1:], cube[0])\n    [([0, -1, 0], -5), ([0, 0, -1], -5), ([-1, 0, 0], -5),\n    ([0, 1, 0], 0), ([1, 0, 0], 0), ([0, 0, 1], 0)]\n    '
    if isinstance(poly, Polygon):
        vertices = list(poly.vertices) + [poly.vertices[0]]
        params = [None] * (len(vertices) - 1)
        for i in range(len(vertices) - 1):
            v1 = vertices[i]
            v2 = vertices[i + 1]
            a1 = v1[1] - v2[1]
            a2 = v2[0] - v1[0]
            b = v2[0] * v1[1] - v2[1] * v1[0]
            factor = gcd_list([a1, a2, b])
            b = S(b) / factor
            a = (S(a1) / factor, S(a2) / factor)
            params[i] = (a, b)
    else:
        params = [None] * len(poly)
        for (i, polygon) in enumerate(poly):
            (v1, v2, v3) = [vertices[vertex] for vertex in polygon[:3]]
            normal = cross_product(v1, v2, v3)
            b = sum([normal[j] * v1[j] for j in range(0, 3)])
            fac = gcd_list(normal)
            if fac.is_zero:
                fac = 1
            normal = [j / fac for j in normal]
            b = b / fac
            params[i] = (normal, b)
    return params

def cross_product(v1, v2, v3):
    if False:
        return 10
    'Returns the cross-product of vectors (v2 - v1) and (v3 - v1)\n    That is : (v2 - v1) X (v3 - v1)\n    '
    v2 = [v2[j] - v1[j] for j in range(0, 3)]
    v3 = [v3[j] - v1[j] for j in range(0, 3)]
    return [v3[2] * v2[1] - v3[1] * v2[2], v3[0] * v2[2] - v3[2] * v2[0], v3[1] * v2[0] - v3[0] * v2[1]]

def best_origin(a, b, lineseg, expr):
    if False:
        for i in range(10):
            print('nop')
    'Helper method for polytope_integrate. Currently not used in the main\n    algorithm.\n\n    Explanation\n    ===========\n\n    Returns a point on the lineseg whose vector inner product with the\n    divergence of `expr` yields an expression with the least maximum\n    total power.\n\n    Parameters\n    ==========\n\n    a :\n        Hyperplane parameter denoting direction.\n    b :\n        Hyperplane parameter denoting distance.\n    lineseg :\n        Line segment on which to find the origin.\n    expr :\n        The expression which determines the best point.\n\n    Algorithm(currently works only for 2D use case)\n    ===============================================\n\n    1 > Firstly, check for edge cases. Here that would refer to vertical\n        or horizontal lines.\n\n    2 > If input expression is a polynomial containing more than one generator\n        then find out the total power of each of the generators.\n\n        x**2 + 3 + x*y + x**4*y**5 ---> {x: 7, y: 6}\n\n        If expression is a constant value then pick the first boundary point\n        of the line segment.\n\n    3 > First check if a point exists on the line segment where the value of\n        the highest power generator becomes 0. If not check if the value of\n        the next highest becomes 0. If none becomes 0 within line segment\n        constraints then pick the first boundary point of the line segment.\n        Actually, any point lying on the segment can be picked as best origin\n        in the last case.\n\n    Examples\n    ========\n\n    >>> from sympy.integrals.intpoly import best_origin\n    >>> from sympy.abc import x, y\n    >>> from sympy import Point, Segment2D\n    >>> l = Segment2D(Point(0, 3), Point(1, 1))\n    >>> expr = x**3*y**7\n    >>> best_origin((2, 1), 3, l, expr)\n    (0, 3.0)\n    '
    (a1, b1) = lineseg.points[0]

    def x_axis_cut(ls):
        if False:
            print('Hello World!')
        'Returns the point where the input line segment\n        intersects the x-axis.\n\n        Parameters\n        ==========\n\n        ls :\n            Line segment\n        '
        (p, q) = ls.points
        if p.y.is_zero:
            return tuple(p)
        elif q.y.is_zero:
            return tuple(q)
        elif p.y / q.y < S.Zero:
            return (p.y * (p.x - q.x) / (q.y - p.y) + p.x, S.Zero)
        else:
            return ()

    def y_axis_cut(ls):
        if False:
            for i in range(10):
                print('nop')
        'Returns the point where the input line segment\n        intersects the y-axis.\n\n        Parameters\n        ==========\n\n        ls :\n            Line segment\n        '
        (p, q) = ls.points
        if p.x.is_zero:
            return tuple(p)
        elif q.x.is_zero:
            return tuple(q)
        elif p.x / q.x < S.Zero:
            return (S.Zero, p.x * (p.y - q.y) / (q.x - p.x) + p.y)
        else:
            return ()
    gens = (x, y)
    power_gens = {}
    for i in gens:
        power_gens[i] = S.Zero
    if len(gens) > 1:
        if len(gens) == 2:
            if a[0] == 0:
                if y_axis_cut(lineseg):
                    return (S.Zero, b / a[1])
                else:
                    return (a1, b1)
            elif a[1] == 0:
                if x_axis_cut(lineseg):
                    return (b / a[0], S.Zero)
                else:
                    return (a1, b1)
        if isinstance(expr, Expr):
            if expr.is_Add:
                for monomial in expr.args:
                    if monomial.is_Pow:
                        if monomial.args[0] in gens:
                            power_gens[monomial.args[0]] += monomial.args[1]
                    else:
                        for univariate in monomial.args:
                            term_type = len(univariate.args)
                            if term_type == 0 and univariate in gens:
                                power_gens[univariate] += 1
                            elif term_type == 2 and univariate.args[0] in gens:
                                power_gens[univariate.args[0]] += univariate.args[1]
            elif expr.is_Mul:
                for term in expr.args:
                    term_type = len(term.args)
                    if term_type == 0 and term in gens:
                        power_gens[term] += 1
                    elif term_type == 2 and term.args[0] in gens:
                        power_gens[term.args[0]] += term.args[1]
            elif expr.is_Pow:
                power_gens[expr.args[0]] = expr.args[1]
            elif expr.is_Symbol:
                power_gens[expr] += 1
        else:
            return (a1, b1)
        power_gens = sorted(power_gens.items(), key=lambda k: str(k[0]))
        if power_gens[0][1] >= power_gens[1][1]:
            if y_axis_cut(lineseg):
                x0 = (S.Zero, b / a[1])
            elif x_axis_cut(lineseg):
                x0 = (b / a[0], S.Zero)
            else:
                x0 = (a1, b1)
        elif x_axis_cut(lineseg):
            x0 = (b / a[0], S.Zero)
        elif y_axis_cut(lineseg):
            x0 = (S.Zero, b / a[1])
        else:
            x0 = (a1, b1)
    else:
        x0 = b / a[0]
    return x0

def decompose(expr, separate=False):
    if False:
        return 10
    'Decomposes an input polynomial into homogeneous ones of\n    smaller or equal degree.\n\n    Explanation\n    ===========\n\n    Returns a dictionary with keys as the degree of the smaller\n    constituting polynomials. Values are the constituting polynomials.\n\n    Parameters\n    ==========\n\n    expr : Expr\n        Polynomial(SymPy expression).\n    separate : bool\n        If True then simply return a list of the constituent monomials\n        If not then break up the polynomial into constituent homogeneous\n        polynomials.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y\n    >>> from sympy.integrals.intpoly import decompose\n    >>> decompose(x**2 + x*y + x + y + x**3*y**2 + y**5)\n    {1: x + y, 2: x**2 + x*y, 5: x**3*y**2 + y**5}\n    >>> decompose(x**2 + x*y + x + y + x**3*y**2 + y**5, True)\n    {x, x**2, y, y**5, x*y, x**3*y**2}\n    '
    poly_dict = {}
    if isinstance(expr, Expr) and (not expr.is_number):
        if expr.is_Symbol:
            poly_dict[1] = expr
        elif expr.is_Add:
            symbols = expr.atoms(Symbol)
            degrees = [(sum(degree_list(monom, *symbols)), monom) for monom in expr.args]
            if separate:
                return {monom[1] for monom in degrees}
            else:
                for monom in degrees:
                    (degree, term) = monom
                    if poly_dict.get(degree):
                        poly_dict[degree] += term
                    else:
                        poly_dict[degree] = term
        elif expr.is_Pow:
            (_, degree) = expr.args
            poly_dict[degree] = expr
        else:
            degree = 0
            for term in expr.args:
                term_type = len(term.args)
                if term_type == 0 and term.is_Symbol:
                    degree += 1
                elif term_type == 2:
                    degree += term.args[1]
            poly_dict[degree] = expr
    else:
        poly_dict[0] = expr
    if separate:
        return set(poly_dict.values())
    return poly_dict

def point_sort(poly, normal=None, clockwise=True):
    if False:
        for i in range(10):
            print('nop')
    "Returns the same polygon with points sorted in clockwise or\n    anti-clockwise order.\n\n    Note that it's necessary for input points to be sorted in some order\n    (clockwise or anti-clockwise) for the integration algorithm to work.\n    As a convention algorithm has been implemented keeping clockwise\n    orientation in mind.\n\n    Parameters\n    ==========\n\n    poly:\n        2D or 3D Polygon.\n    normal : optional\n        The normal of the plane which the 3-Polytope is a part of.\n    clockwise : bool, optional\n        Returns points sorted in clockwise order if True and\n        anti-clockwise if False.\n\n    Examples\n    ========\n\n    >>> from sympy.integrals.intpoly import point_sort\n    >>> from sympy import Point\n    >>> point_sort([Point(0, 0), Point(1, 0), Point(1, 1)])\n    [Point2D(1, 1), Point2D(1, 0), Point2D(0, 0)]\n    "
    pts = poly.vertices if isinstance(poly, Polygon) else poly
    n = len(pts)
    if n < 2:
        return list(pts)
    order = S.One if clockwise else S.NegativeOne
    dim = len(pts[0])
    if dim == 2:
        center = Point(sum((vertex.x for vertex in pts)) / n, sum((vertex.y for vertex in pts)) / n)
    else:
        center = Point(sum((vertex.x for vertex in pts)) / n, sum((vertex.y for vertex in pts)) / n, sum((vertex.z for vertex in pts)) / n)

    def compare(a, b):
        if False:
            while True:
                i = 10
        if a.x - center.x >= S.Zero and b.x - center.x < S.Zero:
            return -order
        elif a.x - center.x < 0 and b.x - center.x >= 0:
            return order
        elif a.x - center.x == 0 and b.x - center.x == 0:
            if a.y - center.y >= 0 or b.y - center.y >= 0:
                return -order if a.y > b.y else order
            return -order if b.y > a.y else order
        det = (a.x - center.x) * (b.y - center.y) - (b.x - center.x) * (a.y - center.y)
        if det < 0:
            return -order
        elif det > 0:
            return order
        first = (a.x - center.x) * (a.x - center.x) + (a.y - center.y) * (a.y - center.y)
        second = (b.x - center.x) * (b.x - center.x) + (b.y - center.y) * (b.y - center.y)
        return -order if first > second else order

    def compare3d(a, b):
        if False:
            i = 10
            return i + 15
        det = cross_product(center, a, b)
        dot_product = sum([det[i] * normal[i] for i in range(0, 3)])
        if dot_product < 0:
            return -order
        elif dot_product > 0:
            return order
    return sorted(pts, key=cmp_to_key(compare if dim == 2 else compare3d))

def norm(point):
    if False:
        for i in range(10):
            print('nop')
    'Returns the Euclidean norm of a point from origin.\n\n    Parameters\n    ==========\n\n    point:\n        This denotes a point in the dimension_al spac_e.\n\n    Examples\n    ========\n\n    >>> from sympy.integrals.intpoly import norm\n    >>> from sympy import Point\n    >>> norm(Point(2, 7))\n    sqrt(53)\n    '
    half = S.Half
    if isinstance(point, (list, tuple)):
        return sum([coord ** 2 for coord in point]) ** half
    elif isinstance(point, Point):
        if isinstance(point, Point2D):
            return (point.x ** 2 + point.y ** 2) ** half
        else:
            return (point.x ** 2 + point.y ** 2 + point.z) ** half
    elif isinstance(point, dict):
        return sum((i ** 2 for i in point.values())) ** half

def intersection(geom_1, geom_2, intersection_type):
    if False:
        while True:
            i = 10
    'Returns intersection between geometric objects.\n\n    Explanation\n    ===========\n\n    Note that this function is meant for use in integration_reduction and\n    at that point in the calling function the lines denoted by the segments\n    surely intersect within segment boundaries. Coincident lines are taken\n    to be non-intersecting. Also, the hyperplane intersection for 2D case is\n    also implemented.\n\n    Parameters\n    ==========\n\n    geom_1, geom_2:\n        The input line segments.\n\n    Examples\n    ========\n\n    >>> from sympy.integrals.intpoly import intersection\n    >>> from sympy import Point, Segment2D\n    >>> l1 = Segment2D(Point(1, 1), Point(3, 5))\n    >>> l2 = Segment2D(Point(2, 0), Point(2, 5))\n    >>> intersection(l1, l2, "segment2D")\n    (2, 3)\n    >>> p1 = ((-1, 0), 0)\n    >>> p2 = ((0, 1), 1)\n    >>> intersection(p1, p2, "plane2D")\n    (0, 1)\n    '
    if intersection_type[:-2] == 'segment':
        if intersection_type == 'segment2D':
            (x1, y1) = geom_1.points[0]
            (x2, y2) = geom_1.points[1]
            (x3, y3) = geom_2.points[0]
            (x4, y4) = geom_2.points[1]
        elif intersection_type == 'segment3D':
            (x1, y1, z1) = geom_1.points[0]
            (x2, y2, z2) = geom_1.points[1]
            (x3, y3, z3) = geom_2.points[0]
            (x4, y4, z4) = geom_2.points[1]
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom:
            t1 = x1 * y2 - y1 * x2
            t2 = x3 * y4 - x4 * y3
            return (S(t1 * (x3 - x4) - t2 * (x1 - x2)) / denom, S(t1 * (y3 - y4) - t2 * (y1 - y2)) / denom)
    if intersection_type[:-2] == 'plane':
        if intersection_type == 'plane2D':
            (a1x, a1y) = geom_1[0]
            (a2x, a2y) = geom_2[0]
            (b1, b2) = (geom_1[1], geom_2[1])
            denom = a1x * a2y - a2x * a1y
            if denom:
                return (S(b1 * a2y - b2 * a1y) / denom, S(b2 * a1x - b1 * a2x) / denom)

def is_vertex(ent):
    if False:
        return 10
    'If the input entity is a vertex return True.\n\n    Parameter\n    =========\n\n    ent :\n        Denotes a geometric entity representing a point.\n\n    Examples\n    ========\n\n    >>> from sympy import Point\n    >>> from sympy.integrals.intpoly import is_vertex\n    >>> is_vertex((2, 3))\n    True\n    >>> is_vertex((2, 3, 6))\n    True\n    >>> is_vertex(Point(2, 3))\n    True\n    '
    if isinstance(ent, tuple):
        if len(ent) in [2, 3]:
            return True
    elif isinstance(ent, Point):
        return True
    return False

def plot_polytope(poly):
    if False:
        i = 10
        return i + 15
    'Plots the 2D polytope using the functions written in plotting\n    module which in turn uses matplotlib backend.\n\n    Parameter\n    =========\n\n    poly:\n        Denotes a 2-Polytope.\n    '
    from sympy.plotting.plot import Plot, List2DSeries
    xl = [vertex.x for vertex in poly.vertices]
    yl = [vertex.y for vertex in poly.vertices]
    xl.append(poly.vertices[0].x)
    yl.append(poly.vertices[0].y)
    l2ds = List2DSeries(xl, yl)
    p = Plot(l2ds, axes='label_axes=True')
    p.show()

def plot_polynomial(expr):
    if False:
        print('Hello World!')
    'Plots the polynomial using the functions written in\n    plotting module which in turn uses matplotlib backend.\n\n    Parameter\n    =========\n\n    expr:\n        Denotes a polynomial(SymPy expression).\n    '
    from sympy.plotting.plot import plot3d, plot
    gens = expr.free_symbols
    if len(gens) == 2:
        plot3d(expr)
    else:
        plot(expr)