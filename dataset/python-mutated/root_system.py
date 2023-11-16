from .cartan_type import CartanType
from sympy.core.basic import Atom

class RootSystem(Atom):
    """Represent the root system of a simple Lie algebra

    Every simple Lie algebra has a unique root system.  To find the root
    system, we first consider the Cartan subalgebra of g, which is the maximal
    abelian subalgebra, and consider the adjoint action of g on this
    subalgebra.  There is a root system associated with this action. Now, a
    root system over a vector space V is a set of finite vectors Phi (called
    roots), which satisfy:

    1.  The roots span V
    2.  The only scalar multiples of x in Phi are x and -x
    3.  For every x in Phi, the set Phi is closed under reflection
        through the hyperplane perpendicular to x.
    4.  If x and y are roots in Phi, then the projection of y onto
        the line through x is a half-integral multiple of x.

    Now, there is a subset of Phi, which we will call Delta, such that:
    1.  Delta is a basis of V
    2.  Each root x in Phi can be written x = sum k_y y for y in Delta

    The elements of Delta are called the simple roots.
    Therefore, we see that the simple roots span the root space of a given
    simple Lie algebra.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Root_system
    .. [2] Lie Algebras and Representation Theory - Humphreys

    """

    def __new__(cls, cartantype):
        if False:
            print('Hello World!')
        'Create a new RootSystem object\n\n        This method assigns an attribute called cartan_type to each instance of\n        a RootSystem object.  When an instance of RootSystem is called, it\n        needs an argument, which should be an instance of a simple Lie algebra.\n        We then take the CartanType of this argument and set it as the\n        cartan_type attribute of the RootSystem instance.\n\n        '
        obj = Atom.__new__(cls)
        obj.cartan_type = CartanType(cartantype)
        return obj

    def simple_roots(self):
        if False:
            for i in range(10):
                print('nop')
        'Generate the simple roots of the Lie algebra\n\n        The rank of the Lie algebra determines the number of simple roots that\n        it has.  This method obtains the rank of the Lie algebra, and then uses\n        the simple_root method from the Lie algebra classes to generate all the\n        simple roots.\n\n        Examples\n        ========\n\n        >>> from sympy.liealgebras.root_system import RootSystem\n        >>> c = RootSystem("A3")\n        >>> roots = c.simple_roots()\n        >>> roots\n        {1: [1, -1, 0, 0], 2: [0, 1, -1, 0], 3: [0, 0, 1, -1]}\n\n        '
        n = self.cartan_type.rank()
        roots = {}
        for i in range(1, n + 1):
            root = self.cartan_type.simple_root(i)
            roots[i] = root
        return roots

    def all_roots(self):
        if False:
            i = 10
            return i + 15
        'Generate all the roots of a given root system\n\n        The result is a dictionary where the keys are integer numbers.  It\n        generates the roots by getting the dictionary of all positive roots\n        from the bases classes, and then taking each root, and multiplying it\n        by -1 and adding it to the dictionary.  In this way all the negative\n        roots are generated.\n\n        '
        alpha = self.cartan_type.positive_roots()
        keys = list(alpha.keys())
        k = max(keys)
        for val in keys:
            k += 1
            root = alpha[val]
            newroot = [-x for x in root]
            alpha[k] = newroot
        return alpha

    def root_space(self):
        if False:
            i = 10
            return i + 15
        'Return the span of the simple roots\n\n        The root space is the vector space spanned by the simple roots, i.e. it\n        is a vector space with a distinguished basis, the simple roots.  This\n        method returns a string that represents the root space as the span of\n        the simple roots, alpha[1],...., alpha[n].\n\n        Examples\n        ========\n\n        >>> from sympy.liealgebras.root_system import RootSystem\n        >>> c = RootSystem("A3")\n        >>> c.root_space()\n        \'alpha[1] + alpha[2] + alpha[3]\'\n\n        '
        n = self.cartan_type.rank()
        rs = ' + '.join(('alpha[' + str(i) + ']' for i in range(1, n + 1)))
        return rs

    def add_simple_roots(self, root1, root2):
        if False:
            print('Hello World!')
        'Add two simple roots together\n\n        The function takes as input two integers, root1 and root2.  It then\n        uses these integers as keys in the dictionary of simple roots, and gets\n        the corresponding simple roots, and then adds them together.\n\n        Examples\n        ========\n\n        >>> from sympy.liealgebras.root_system import RootSystem\n        >>> c = RootSystem("A3")\n        >>> newroot = c.add_simple_roots(1, 2)\n        >>> newroot\n        [1, 0, -1, 0]\n\n        '
        alpha = self.simple_roots()
        if root1 > len(alpha) or root2 > len(alpha):
            raise ValueError("You've used a root that doesn't exist!")
        a1 = alpha[root1]
        a2 = alpha[root2]
        newroot = [_a1 + _a2 for (_a1, _a2) in zip(a1, a2)]
        return newroot

    def add_as_roots(self, root1, root2):
        if False:
            print('Hello World!')
        'Add two roots together if and only if their sum is also a root\n\n        It takes as input two vectors which should be roots.  It then computes\n        their sum and checks if it is in the list of all possible roots.  If it\n        is, it returns the sum.  Otherwise it returns a string saying that the\n        sum is not a root.\n\n        Examples\n        ========\n\n        >>> from sympy.liealgebras.root_system import RootSystem\n        >>> c = RootSystem("A3")\n        >>> c.add_as_roots([1, 0, -1, 0], [0, 0, 1, -1])\n        [1, 0, 0, -1]\n        >>> c.add_as_roots([1, -1, 0, 0], [0, 0, -1, 1])\n        \'The sum of these two roots is not a root\'\n\n        '
        alpha = self.all_roots()
        newroot = [r1 + r2 for (r1, r2) in zip(root1, root2)]
        if newroot in alpha.values():
            return newroot
        else:
            return 'The sum of these two roots is not a root'

    def cartan_matrix(self):
        if False:
            print('Hello World!')
        'Cartan matrix of Lie algebra associated with this root system\n\n        Examples\n        ========\n\n        >>> from sympy.liealgebras.root_system import RootSystem\n        >>> c = RootSystem("A3")\n        >>> c.cartan_matrix()\n        Matrix([\n            [ 2, -1,  0],\n            [-1,  2, -1],\n            [ 0, -1,  2]])\n        '
        return self.cartan_type.cartan_matrix()

    def dynkin_diagram(self):
        if False:
            print('Hello World!')
        'Dynkin diagram of the Lie algebra associated with this root system\n\n        Examples\n        ========\n\n        >>> from sympy.liealgebras.root_system import RootSystem\n        >>> c = RootSystem("A3")\n        >>> print(c.dynkin_diagram())\n        0---0---0\n        1   2   3\n        '
        return self.cartan_type.dynkin_diagram()