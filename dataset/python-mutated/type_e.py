from .cartan_type import Standard_Cartan
from sympy.core.backend import eye, Rational

class TypeE(Standard_Cartan):

    def __new__(cls, n):
        if False:
            return 10
        if n < 6 or n > 8:
            raise ValueError('Invalid value of n')
        return Standard_Cartan.__new__(cls, 'E', n)

    def dimension(self):
        if False:
            i = 10
            return i + 15
        'Dimension of the vector space V underlying the Lie algebra\n\n        Examples\n        ========\n\n        >>> from sympy.liealgebras.cartan_type import CartanType\n        >>> c = CartanType("E6")\n        >>> c.dimension()\n        8\n        '
        return 8

    def basic_root(self, i, j):
        if False:
            i = 10
            return i + 15
        '\n        This is a method just to generate roots\n        with a -1 in the ith position and a 1\n        in the jth position.\n\n        '
        root = [0] * 8
        root[i] = -1
        root[j] = 1
        return root

    def simple_root(self, i):
        if False:
            for i in range(10):
                print('nop')
        '\n        Every lie algebra has a unique root system.\n        Given a root system Q, there is a subset of the\n        roots such that an element of Q is called a\n        simple root if it cannot be written as the sum\n        of two elements in Q.  If we let D denote the\n        set of simple roots, then it is clear that every\n        element of Q can be written as a linear combination\n        of elements of D with all coefficients non-negative.\n\n        This method returns the ith simple root for E_n.\n\n        Examples\n        ========\n\n        >>> from sympy.liealgebras.cartan_type import CartanType\n        >>> c = CartanType("E6")\n        >>> c.simple_root(2)\n        [1, 1, 0, 0, 0, 0, 0, 0]\n        '
        n = self.n
        if i == 1:
            root = [-0.5] * 8
            root[0] = 0.5
            root[7] = 0.5
            return root
        elif i == 2:
            root = [0] * 8
            root[1] = 1
            root[0] = 1
            return root
        else:
            if i in (7, 8) and n == 6:
                raise ValueError('E6 only has six simple roots!')
            if i == 8 and n == 7:
                raise ValueError('E7 has only 7 simple roots!')
            return self.basic_root(i - 3, i - 2)

    def positive_roots(self):
        if False:
            return 10
        '\n        This method generates all the positive roots of\n        A_n.  This is half of all of the roots of E_n;\n        by multiplying all the positive roots by -1 we\n        get the negative roots.\n\n        Examples\n        ========\n\n        >>> from sympy.liealgebras.cartan_type import CartanType\n        >>> c = CartanType("A3")\n        >>> c.positive_roots()\n        {1: [1, -1, 0, 0], 2: [1, 0, -1, 0], 3: [1, 0, 0, -1], 4: [0, 1, -1, 0],\n                5: [0, 1, 0, -1], 6: [0, 0, 1, -1]}\n        '
        n = self.n
        if n == 6:
            posroots = {}
            k = 0
            for i in range(n - 1):
                for j in range(i + 1, n - 1):
                    k += 1
                    root = self.basic_root(i, j)
                    posroots[k] = root
                    k += 1
                    root = self.basic_root(i, j)
                    root[i] = 1
                    posroots[k] = root
            root = [Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(-1, 2), Rational(-1, 2), Rational(1, 2)]
            for a in range(0, 2):
                for b in range(0, 2):
                    for c in range(0, 2):
                        for d in range(0, 2):
                            for e in range(0, 2):
                                if (a + b + c + d + e) % 2 == 0:
                                    k += 1
                                    if a == 1:
                                        root[0] = Rational(-1, 2)
                                    if b == 1:
                                        root[1] = Rational(-1, 2)
                                    if c == 1:
                                        root[2] = Rational(-1, 2)
                                    if d == 1:
                                        root[3] = Rational(-1, 2)
                                    if e == 1:
                                        root[4] = Rational(-1, 2)
                                    posroots[k] = root
            return posroots
        if n == 7:
            posroots = {}
            k = 0
            for i in range(n - 1):
                for j in range(i + 1, n - 1):
                    k += 1
                    root = self.basic_root(i, j)
                    posroots[k] = root
                    k += 1
                    root = self.basic_root(i, j)
                    root[i] = 1
                    posroots[k] = root
            k += 1
            posroots[k] = [0, 0, 0, 0, 0, 1, 1, 0]
            root = [Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(-1, 2), Rational(-1, 2), Rational(1, 2)]
            for a in range(0, 2):
                for b in range(0, 2):
                    for c in range(0, 2):
                        for d in range(0, 2):
                            for e in range(0, 2):
                                for f in range(0, 2):
                                    if (a + b + c + d + e + f) % 2 == 0:
                                        k += 1
                                        if a == 1:
                                            root[0] = Rational(-1, 2)
                                        if b == 1:
                                            root[1] = Rational(-1, 2)
                                        if c == 1:
                                            root[2] = Rational(-1, 2)
                                        if d == 1:
                                            root[3] = Rational(-1, 2)
                                        if e == 1:
                                            root[4] = Rational(-1, 2)
                                        if f == 1:
                                            root[5] = Rational(1, 2)
                                        posroots[k] = root
            return posroots
        if n == 8:
            posroots = {}
            k = 0
            for i in range(n):
                for j in range(i + 1, n):
                    k += 1
                    root = self.basic_root(i, j)
                    posroots[k] = root
                    k += 1
                    root = self.basic_root(i, j)
                    root[i] = 1
                    posroots[k] = root
            root = [Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(-1, 2), Rational(-1, 2), Rational(1, 2)]
            for a in range(0, 2):
                for b in range(0, 2):
                    for c in range(0, 2):
                        for d in range(0, 2):
                            for e in range(0, 2):
                                for f in range(0, 2):
                                    for g in range(0, 2):
                                        if (a + b + c + d + e + f + g) % 2 == 0:
                                            k += 1
                                            if a == 1:
                                                root[0] = Rational(-1, 2)
                                            if b == 1:
                                                root[1] = Rational(-1, 2)
                                            if c == 1:
                                                root[2] = Rational(-1, 2)
                                            if d == 1:
                                                root[3] = Rational(-1, 2)
                                            if e == 1:
                                                root[4] = Rational(-1, 2)
                                            if f == 1:
                                                root[5] = Rational(1, 2)
                                            if g == 1:
                                                root[6] = Rational(1, 2)
                                            posroots[k] = root
            return posroots

    def roots(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the total number of roots of E_n\n        '
        n = self.n
        if n == 6:
            return 72
        if n == 7:
            return 126
        if n == 8:
            return 240

    def cartan_matrix(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns the Cartan matrix for G_2\n        The Cartan matrix matrix for a Lie algebra is\n        generated by assigning an ordering to the simple\n        roots, (alpha[1], ...., alpha[l]).  Then the ijth\n        entry of the Cartan matrix is (<alpha[i],alpha[j]>).\n\n        Examples\n        ========\n\n        >>> from sympy.liealgebras.cartan_type import CartanType\n        >>> c = CartanType('A4')\n        >>> c.cartan_matrix()\n        Matrix([\n        [ 2, -1,  0,  0],\n        [-1,  2, -1,  0],\n        [ 0, -1,  2, -1],\n        [ 0,  0, -1,  2]])\n\n\n        "
        n = self.n
        m = 2 * eye(n)
        i = 3
        while i < n - 1:
            m[i, i + 1] = -1
            m[i, i - 1] = -1
            i += 1
        m[0, 2] = m[2, 0] = -1
        m[1, 3] = m[3, 1] = -1
        m[2, 3] = -1
        m[n - 1, n - 2] = -1
        return m

    def basis(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of independent generators of E_n\n        '
        n = self.n
        if n == 6:
            return 78
        if n == 7:
            return 133
        if n == 8:
            return 248

    def dynkin_diagram(self):
        if False:
            return 10
        n = self.n
        diag = ' ' * 8 + str(2) + '\n'
        diag += ' ' * 8 + '0\n'
        diag += ' ' * 8 + '|\n'
        diag += ' ' * 8 + '|\n'
        diag += '---'.join(('0' for i in range(1, n))) + '\n'
        diag += '1   ' + '   '.join((str(i) for i in range(3, n + 1)))
        return diag