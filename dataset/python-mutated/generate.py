"""
Generating and counting primes.

"""
from bisect import bisect, bisect_left
from itertools import count
from array import array as _array
from sympy.core.function import Function
from sympy.core.random import randint
from sympy.core.singleton import S
from sympy.external.gmpy import sqrt
from .primetest import isprime
from sympy.utilities.misc import as_int

def _as_int_ceiling(a):
    if False:
        print('Hello World!')
    ' Wrapping ceiling in as_int will raise an error if there was a problem\n        determining whether the expression was exactly an integer or not.'
    from sympy.functions.elementary.integers import ceiling
    return as_int(ceiling(a))

class Sieve:
    """A list of prime numbers, implemented as a dynamically
    growing sieve of Eratosthenes. When a lookup is requested involving
    an odd number that has not been sieved, the sieve is automatically
    extended up to that number. Implementation details limit the number of
    primes to ``2^32-1``.

    Examples
    ========

    >>> from sympy import sieve
    >>> sieve._reset() # this line for doctest only
    >>> 25 in sieve
    False
    >>> sieve._list
    array('L', [2, 3, 5, 7, 11, 13, 17, 19, 23])
    """

    def __init__(self, sieve_interval=1000000):
        if False:
            i = 10
            return i + 15
        ' Initial parameters for the Sieve class.\n\n        Parameters\n        ==========\n\n        sieve_interval (int): Amount of memory to be used\n\n        Raises\n        ======\n\n        ValueError\n            If ``sieve_interval`` is not positive.\n\n        '
        self._n = 6
        self._list = _array('L', [2, 3, 5, 7, 11, 13])
        self._tlist = _array('L', [0, 1, 1, 2, 2, 4])
        self._mlist = _array('i', [0, 1, -1, -1, 0, -1])
        if sieve_interval <= 0:
            raise ValueError('sieve_interval should be a positive integer')
        self.sieve_interval = sieve_interval
        assert all((len(i) == self._n for i in (self._list, self._tlist, self._mlist)))

    def __repr__(self):
        if False:
            return 10
        return '<%s sieve (%i): %i, %i, %i, ... %i, %i\n%s sieve (%i): %i, %i, %i, ... %i, %i\n%s sieve (%i): %i, %i, %i, ... %i, %i>' % ('prime', len(self._list), self._list[0], self._list[1], self._list[2], self._list[-2], self._list[-1], 'totient', len(self._tlist), self._tlist[0], self._tlist[1], self._tlist[2], self._tlist[-2], self._tlist[-1], 'mobius', len(self._mlist), self._mlist[0], self._mlist[1], self._mlist[2], self._mlist[-2], self._mlist[-1])

    def _reset(self, prime=None, totient=None, mobius=None):
        if False:
            return 10
        'Reset all caches (default). To reset one or more set the\n            desired keyword to True.'
        if all((i is None for i in (prime, totient, mobius))):
            prime = totient = mobius = True
        if prime:
            self._list = self._list[:self._n]
        if totient:
            self._tlist = self._tlist[:self._n]
        if mobius:
            self._mlist = self._mlist[:self._n]

    def extend(self, n):
        if False:
            for i in range(10):
                print('nop')
        'Grow the sieve to cover all primes <= n.\n\n        Examples\n        ========\n\n        >>> from sympy import sieve\n        >>> sieve._reset() # this line for doctest only\n        >>> sieve.extend(30)\n        >>> sieve[10] == 29\n        True\n        '
        n = int(n)
        num = self._list[-1] + 1
        if n < num:
            return
        num2 = num ** 2
        while num2 <= n:
            self._list += _array('L', self._primerange(num, num2))
            (num, num2) = (num2, num2 ** 2)
        self._list += _array('L', self._primerange(num, n + 1))

    def _primerange(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        ' Generate all prime numbers in the range (a, b).\n\n        Parameters\n        ==========\n\n        a, b : positive integers assuming the following conditions\n                * a is an even number\n                * 2 < self._list[-1] < a < b < nextprime(self._list[-1])**2\n\n        Yields\n        ======\n\n        p (int): prime numbers such that ``a < p < b``\n\n        Examples\n        ========\n\n        >>> from sympy.ntheory.generate import Sieve\n        >>> s = Sieve()\n        >>> s._list[-1]\n        13\n        >>> list(s._primerange(18, 31))\n        [19, 23, 29]\n\n        '
        if b % 2:
            b -= 1
        while a < b:
            block_size = min(self.sieve_interval, (b - a) // 2)
            block = [True] * block_size
            for p in self._list[1:bisect(self._list, sqrt(a + 2 * block_size + 1))]:
                for t in range(-(a + 1 + p) // 2 % p, block_size, p):
                    block[t] = False
            for (idx, p) in enumerate(block):
                if p:
                    yield (a + 2 * idx + 1)
            a += 2 * block_size

    def extend_to_no(self, i):
        if False:
            for i in range(10):
                print('nop')
        "Extend to include the ith prime number.\n\n        Parameters\n        ==========\n\n        i : integer\n\n        Examples\n        ========\n\n        >>> from sympy import sieve\n        >>> sieve._reset() # this line for doctest only\n        >>> sieve.extend_to_no(9)\n        >>> sieve._list\n        array('L', [2, 3, 5, 7, 11, 13, 17, 19, 23])\n\n        Notes\n        =====\n\n        The list is extended by 50% if it is too short, so it is\n        likely that it will be longer than requested.\n        "
        i = as_int(i)
        while len(self._list) < i:
            self.extend(int(self._list[-1] * 1.5))

    def primerange(self, a, b=None):
        if False:
            for i in range(10):
                print('nop')
        'Generate all prime numbers in the range [2, a) or [a, b).\n\n        Examples\n        ========\n\n        >>> from sympy import sieve, prime\n\n        All primes less than 19:\n\n        >>> print([i for i in sieve.primerange(19)])\n        [2, 3, 5, 7, 11, 13, 17]\n\n        All primes greater than or equal to 7 and less than 19:\n\n        >>> print([i for i in sieve.primerange(7, 19)])\n        [7, 11, 13, 17]\n\n        All primes through the 10th prime\n\n        >>> list(sieve.primerange(prime(10) + 1))\n        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]\n\n        '
        if b is None:
            b = _as_int_ceiling(a)
            a = 2
        else:
            a = max(2, _as_int_ceiling(a))
            b = _as_int_ceiling(b)
        if a >= b:
            return
        self.extend(b)
        yield from self._list[bisect_left(self._list, a):bisect_left(self._list, b)]

    def totientrange(self, a, b):
        if False:
            return 10
        'Generate all totient numbers for the range [a, b).\n\n        Examples\n        ========\n\n        >>> from sympy import sieve\n        >>> print([i for i in sieve.totientrange(7, 18)])\n        [6, 4, 6, 4, 10, 4, 12, 6, 8, 8, 16]\n        '
        a = max(1, _as_int_ceiling(a))
        b = _as_int_ceiling(b)
        n = len(self._tlist)
        if a >= b:
            return
        elif b <= n:
            for i in range(a, b):
                yield self._tlist[i]
        else:
            self._tlist += _array('L', range(n, b))
            for i in range(1, n):
                ti = self._tlist[i]
                if ti == i - 1:
                    startindex = (n + i - 1) // i * i
                    for j in range(startindex, b, i):
                        self._tlist[j] -= self._tlist[j] // i
                if i >= a:
                    yield ti
            for i in range(n, b):
                ti = self._tlist[i]
                if ti == i:
                    for j in range(i, b, i):
                        self._tlist[j] -= self._tlist[j] // i
                if i >= a:
                    yield self._tlist[i]

    def mobiusrange(self, a, b):
        if False:
            print('Hello World!')
        'Generate all mobius numbers for the range [a, b).\n\n        Parameters\n        ==========\n\n        a : integer\n            First number in range\n\n        b : integer\n            First number outside of range\n\n        Examples\n        ========\n\n        >>> from sympy import sieve\n        >>> print([i for i in sieve.mobiusrange(7, 18)])\n        [-1, 0, 0, 1, -1, 0, -1, 1, 1, 0, -1]\n        '
        a = max(1, _as_int_ceiling(a))
        b = _as_int_ceiling(b)
        n = len(self._mlist)
        if a >= b:
            return
        elif b <= n:
            for i in range(a, b):
                yield self._mlist[i]
        else:
            self._mlist += _array('i', [0] * (b - n))
            for i in range(1, n):
                mi = self._mlist[i]
                startindex = (n + i - 1) // i * i
                for j in range(startindex, b, i):
                    self._mlist[j] -= mi
                if i >= a:
                    yield mi
            for i in range(n, b):
                mi = self._mlist[i]
                for j in range(2 * i, b, i):
                    self._mlist[j] -= mi
                if i >= a:
                    yield mi

    def search(self, n):
        if False:
            return 10
        'Return the indices i, j of the primes that bound n.\n\n        If n is prime then i == j.\n\n        Although n can be an expression, if ceiling cannot convert\n        it to an integer then an n error will be raised.\n\n        Examples\n        ========\n\n        >>> from sympy import sieve\n        >>> sieve.search(25)\n        (9, 10)\n        >>> sieve.search(23)\n        (9, 9)\n        '
        test = _as_int_ceiling(n)
        n = as_int(n)
        if n < 2:
            raise ValueError('n should be >= 2 but got: %s' % n)
        if n > self._list[-1]:
            self.extend(n)
        b = bisect(self._list, n)
        if self._list[b - 1] == test:
            return (b, b)
        else:
            return (b, b + 1)

    def __contains__(self, n):
        if False:
            print('Hello World!')
        try:
            n = as_int(n)
            assert n >= 2
        except (ValueError, AssertionError):
            return False
        if n % 2 == 0:
            return n == 2
        (a, b) = self.search(n)
        return a == b

    def __iter__(self):
        if False:
            return 10
        for n in count(1):
            yield self[n]

    def __getitem__(self, n):
        if False:
            i = 10
            return i + 15
        'Return the nth prime number'
        if isinstance(n, slice):
            self.extend_to_no(n.stop)
            start = n.start if n.start is not None else 0
            if start < 1:
                raise IndexError('Sieve indices start at 1.')
            return self._list[start - 1:n.stop - 1:n.step]
        else:
            if n < 1:
                raise IndexError('Sieve indices start at 1.')
            n = as_int(n)
            self.extend_to_no(n)
            return self._list[n - 1]
sieve = Sieve()

def prime(nth):
    if False:
        i = 10
        return i + 15
    ' Return the nth prime, with the primes indexed as prime(1) = 2,\n        prime(2) = 3, etc.... The nth prime is approximately $n\\log(n)$.\n\n        Logarithmic integral of $x$ is a pretty nice approximation for number of\n        primes $\\le x$, i.e.\n        li(x) ~ pi(x)\n        In fact, for the numbers we are concerned about( x<1e11 ),\n        li(x) - pi(x) < 50000\n\n        Also,\n        li(x) > pi(x) can be safely assumed for the numbers which\n        can be evaluated by this function.\n\n        Here, we find the least integer m such that li(m) > n using binary search.\n        Now pi(m-1) < li(m-1) <= n,\n\n        We find pi(m - 1) using primepi function.\n\n        Starting from m, we have to find n - pi(m-1) more primes.\n\n        For the inputs this implementation can handle, we will have to test\n        primality for at max about 10**5 numbers, to get our answer.\n\n        Examples\n        ========\n\n        >>> from sympy import prime\n        >>> prime(10)\n        29\n        >>> prime(1)\n        2\n        >>> prime(100000)\n        1299709\n\n        See Also\n        ========\n\n        sympy.ntheory.primetest.isprime : Test if n is prime\n        primerange : Generate all primes in a given range\n        primepi : Return the number of primes less than or equal to n\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Prime_number_theorem#Table_of_.CF.80.28x.29.2C_x_.2F_log_x.2C_and_li.28x.29\n        .. [2] https://en.wikipedia.org/wiki/Prime_number_theorem#Approximations_for_the_nth_prime_number\n        .. [3] https://en.wikipedia.org/wiki/Skewes%27_number\n    '
    n = as_int(nth)
    if n < 1:
        raise ValueError('nth must be a positive integer; prime(1) == 2')
    if n <= len(sieve._list):
        return sieve[n]
    from sympy.functions.elementary.exponential import log
    from sympy.functions.special.error_functions import li
    a = 2
    b = int(n * (log(n) + log(log(n))))
    while a < b:
        mid = a + b >> 1
        if li(mid) > n:
            b = mid
        else:
            a = mid + 1
    n_primes = primepi(a - 1)
    while n_primes < n:
        if isprime(a):
            n_primes += 1
        a += 1
    return a - 1

class primepi(Function):
    """ Represents the prime counting function pi(n) = the number
        of prime numbers less than or equal to n.

        Algorithm Description:

        In sieve method, we remove all multiples of prime p
        except p itself.

        Let phi(i,j) be the number of integers 2 <= k <= i
        which remain after sieving from primes less than
        or equal to j.
        Clearly, pi(n) = phi(n, sqrt(n))

        If j is not a prime,
        phi(i,j) = phi(i, j - 1)

        if j is a prime,
        We remove all numbers(except j) whose
        smallest prime factor is j.

        Let $x= j \\times a$ be such a number, where $2 \\le a \\le i / j$
        Now, after sieving from primes $\\le j - 1$,
        a must remain
        (because x, and hence a has no prime factor $\\le j - 1$)
        Clearly, there are phi(i / j, j - 1) such a
        which remain on sieving from primes $\\le j - 1$

        Now, if a is a prime less than equal to j - 1,
        $x= j \\times a$ has smallest prime factor = a, and
        has already been removed(by sieving from a).
        So, we do not need to remove it again.
        (Note: there will be pi(j - 1) such x)

        Thus, number of x, that will be removed are:
        phi(i / j, j - 1) - phi(j - 1, j - 1)
        (Note that pi(j - 1) = phi(j - 1, j - 1))

        $\\Rightarrow$ phi(i,j) = phi(i, j - 1) - phi(i / j, j - 1) + phi(j - 1, j - 1)

        So,following recursion is used and implemented as dp:

        phi(a, b) = phi(a, b - 1), if b is not a prime
        phi(a, b) = phi(a, b-1)-phi(a / b, b-1) + phi(b-1, b-1), if b is prime

        Clearly a is always of the form floor(n / k),
        which can take at most $2\\sqrt{n}$ values.
        Two arrays arr1,arr2 are maintained
        arr1[i] = phi(i, j),
        arr2[i] = phi(n // i, j)

        Finally the answer is arr2[1]

        Examples
        ========

        >>> from sympy import primepi, prime, prevprime, isprime
        >>> primepi(25)
        9

        So there are 9 primes less than or equal to 25. Is 25 prime?

        >>> isprime(25)
        False

        It is not. So the first prime less than 25 must be the
        9th prime:

        >>> prevprime(25) == prime(9)
        True

        See Also
        ========

        sympy.ntheory.primetest.isprime : Test if n is prime
        primerange : Generate all primes in a given range
        prime : Return the nth prime
    """

    @classmethod
    def eval(cls, n):
        if False:
            return 10
        if n is S.Infinity:
            return S.Infinity
        if n is S.NegativeInfinity:
            return S.Zero
        try:
            n = int(n)
        except TypeError:
            if n.is_real == False or n is S.NaN:
                raise ValueError('n must be real')
            return
        if n < 2:
            return S.Zero
        if n <= sieve._list[-1]:
            return S(sieve.search(n)[0])
        lim = int(n ** 0.5)
        lim -= 1
        lim = max(lim, 0)
        while lim * lim <= n:
            lim += 1
        lim -= 1
        arr1 = [0] * (lim + 1)
        arr2 = [0] * (lim + 1)
        for i in range(1, lim + 1):
            arr1[i] = i - 1
            arr2[i] = n // i - 1
        for i in range(2, lim + 1):
            if arr1[i] == arr1[i - 1]:
                continue
            p = arr1[i - 1]
            for j in range(1, min(n // (i * i), lim) + 1):
                st = i * j
                if st <= lim:
                    arr2[j] -= arr2[st] - p
                else:
                    arr2[j] -= arr1[n // st] - p
            lim2 = min(lim, i * i - 1)
            for j in range(lim, lim2, -1):
                arr1[j] -= arr1[j // i] - p
        return S(arr2[1])

def nextprime(n, ith=1):
    if False:
        for i in range(10):
            print('nop')
    ' Return the ith prime greater than n.\n\n        Parameters\n        ==========\n\n        n : integer\n        ith : positive integer\n\n        Returns\n        =======\n\n        int : Return the ith prime greater than n\n\n        Raises\n        ======\n\n        ValueError\n            If ``ith <= 0``.\n            If ``n`` or ``ith`` is not an integer.\n\n        Notes\n        =====\n\n        Potential primes are located at 6*j +/- 1. This\n        property is used during searching.\n\n        >>> from sympy import nextprime\n        >>> [(i, nextprime(i)) for i in range(10, 15)]\n        [(10, 11), (11, 13), (12, 13), (13, 17), (14, 17)]\n        >>> nextprime(2, ith=2) # the 2nd prime after 2\n        5\n\n        See Also\n        ========\n\n        prevprime : Return the largest prime smaller than n\n        primerange : Generate all primes in a given range\n\n    '
    n = int(n)
    i = as_int(ith)
    if i <= 0:
        raise ValueError('ith should be positive')
    if n < 2:
        n = 2
        i -= 1
    if n <= sieve._list[-2]:
        (l, _) = sieve.search(n)
        if l + i - 1 < len(sieve._list):
            return sieve._list[l + i - 1]
        return nextprime(sieve._list[-1], l + i - len(sieve._list))
    if 1 < i:
        for _ in range(i):
            n = nextprime(n)
        return n
    nn = 6 * (n // 6)
    if nn == n:
        n += 1
        if isprime(n):
            return n
        n += 4
    elif n - nn == 5:
        n += 2
        if isprime(n):
            return n
        n += 4
    else:
        n = nn + 5
    while 1:
        if isprime(n):
            return n
        n += 2
        if isprime(n):
            return n
        n += 4

def prevprime(n):
    if False:
        for i in range(10):
            print('nop')
    ' Return the largest prime smaller than n.\n\n        Notes\n        =====\n\n        Potential primes are located at 6*j +/- 1. This\n        property is used during searching.\n\n        >>> from sympy import prevprime\n        >>> [(i, prevprime(i)) for i in range(10, 15)]\n        [(10, 7), (11, 7), (12, 11), (13, 11), (14, 13)]\n\n        See Also\n        ========\n\n        nextprime : Return the ith prime greater than n\n        primerange : Generates all primes in a given range\n    '
    n = _as_int_ceiling(n)
    if n < 3:
        raise ValueError('no preceding primes')
    if n < 8:
        return {3: 2, 4: 3, 5: 3, 6: 5, 7: 5}[n]
    if n <= sieve._list[-1]:
        (l, u) = sieve.search(n)
        if l == u:
            return sieve[l - 1]
        else:
            return sieve[l]
    nn = 6 * (n // 6)
    if n - nn <= 1:
        n = nn - 1
        if isprime(n):
            return n
        n -= 4
    else:
        n = nn + 1
    while 1:
        if isprime(n):
            return n
        n -= 2
        if isprime(n):
            return n
        n -= 4

def primerange(a, b=None):
    if False:
        return 10
    " Generate a list of all prime numbers in the range [2, a),\n        or [a, b).\n\n        If the range exists in the default sieve, the values will\n        be returned from there; otherwise values will be returned\n        but will not modify the sieve.\n\n        Examples\n        ========\n\n        >>> from sympy import primerange, prime\n\n        All primes less than 19:\n\n        >>> list(primerange(19))\n        [2, 3, 5, 7, 11, 13, 17]\n\n        All primes greater than or equal to 7 and less than 19:\n\n        >>> list(primerange(7, 19))\n        [7, 11, 13, 17]\n\n        All primes through the 10th prime\n\n        >>> list(primerange(prime(10) + 1))\n        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]\n\n        The Sieve method, primerange, is generally faster but it will\n        occupy more memory as the sieve stores values. The default\n        instance of Sieve, named sieve, can be used:\n\n        >>> from sympy import sieve\n        >>> list(sieve.primerange(1, 30))\n        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]\n\n        Notes\n        =====\n\n        Some famous conjectures about the occurrence of primes in a given\n        range are [1]:\n\n        - Twin primes: though often not, the following will give 2 primes\n                    an infinite number of times:\n                        primerange(6*n - 1, 6*n + 2)\n        - Legendre's: the following always yields at least one prime\n                        primerange(n**2, (n+1)**2+1)\n        - Bertrand's (proven): there is always a prime in the range\n                        primerange(n, 2*n)\n        - Brocard's: there are at least four primes in the range\n                        primerange(prime(n)**2, prime(n+1)**2)\n\n        The average gap between primes is log(n) [2]; the gap between\n        primes can be arbitrarily large since sequences of composite\n        numbers are arbitrarily large, e.g. the numbers in the sequence\n        n! + 2, n! + 3 ... n! + n are all composite.\n\n        See Also\n        ========\n\n        prime : Return the nth prime\n        nextprime : Return the ith prime greater than n\n        prevprime : Return the largest prime smaller than n\n        randprime : Returns a random prime in a given range\n        primorial : Returns the product of primes based on condition\n        Sieve.primerange : return range from already computed primes\n                           or extend the sieve to contain the requested\n                           range.\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Prime_number\n        .. [2] https://primes.utm.edu/notes/gaps.html\n    "
    if b is None:
        (a, b) = (2, a)
    if a >= b:
        return
    largest_known_prime = sieve._list[-1]
    if b <= largest_known_prime:
        yield from sieve.primerange(a, b)
        return
    if a <= largest_known_prime:
        yield from sieve._list[bisect_left(sieve._list, a):]
        a = largest_known_prime + 1
    elif a % 2:
        a -= 1
    tail = min(b, largest_known_prime ** 2)
    if a < tail:
        yield from sieve._primerange(a, tail)
        a = tail
    if b <= a:
        return
    while 1:
        a = nextprime(a)
        if a < b:
            yield a
        else:
            return

def randprime(a, b):
    if False:
        return 10
    " Return a random prime number in the range [a, b).\n\n        Bertrand's postulate assures that\n        randprime(a, 2*a) will always succeed for a > 1.\n\n        Note that due to implementation difficulties,\n        the prime numbers chosen are not uniformly random.\n        For example, there are two primes in the range [112, 128),\n        ``113`` and ``127``, but ``randprime(112, 128)`` returns ``127``\n        with a probability of 15/17.\n\n        Examples\n        ========\n\n        >>> from sympy import randprime, isprime\n        >>> randprime(1, 30) #doctest: +SKIP\n        13\n        >>> isprime(randprime(1, 30))\n        True\n\n        See Also\n        ========\n\n        primerange : Generate all primes in a given range\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Bertrand's_postulate\n\n    "
    if a >= b:
        return
    (a, b) = map(int, (a, b))
    n = randint(a - 1, b)
    p = nextprime(n)
    if p >= b:
        p = prevprime(b)
    if p < a:
        raise ValueError('no primes exist in the specified range')
    return p

def primorial(n, nth=True):
    if False:
        while True:
            i = 10
    '\n    Returns the product of the first n primes (default) or\n    the primes less than or equal to n (when ``nth=False``).\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.generate import primorial, primerange\n    >>> from sympy import factorint, Mul, primefactors, sqrt\n    >>> primorial(4) # the first 4 primes are 2, 3, 5, 7\n    210\n    >>> primorial(4, nth=False) # primes <= 4 are 2 and 3\n    6\n    >>> primorial(1)\n    2\n    >>> primorial(1, nth=False)\n    1\n    >>> primorial(sqrt(101), nth=False)\n    210\n\n    One can argue that the primes are infinite since if you take\n    a set of primes and multiply them together (e.g. the primorial) and\n    then add or subtract 1, the result cannot be divided by any of the\n    original factors, hence either 1 or more new primes must divide this\n    product of primes.\n\n    In this case, the number itself is a new prime:\n\n    >>> factorint(primorial(4) + 1)\n    {211: 1}\n\n    In this case two new primes are the factors:\n\n    >>> factorint(primorial(4) - 1)\n    {11: 1, 19: 1}\n\n    Here, some primes smaller and larger than the primes multiplied together\n    are obtained:\n\n    >>> p = list(primerange(10, 20))\n    >>> sorted(set(primefactors(Mul(*p) + 1)).difference(set(p)))\n    [2, 5, 31, 149]\n\n    See Also\n    ========\n\n    primerange : Generate all primes in a given range\n\n    '
    if nth:
        n = as_int(n)
    else:
        n = int(n)
    if n < 1:
        raise ValueError('primorial argument must be >= 1')
    p = 1
    if nth:
        for i in range(1, n + 1):
            p *= prime(i)
    else:
        for i in primerange(2, n + 1):
            p *= i
    return p

def cycle_length(f, x0, nmax=None, values=False):
    if False:
        print('Hello World!')
    "For a given iterated sequence, return a generator that gives\n    the length of the iterated cycle (lambda) and the length of terms\n    before the cycle begins (mu); if ``values`` is True then the\n    terms of the sequence will be returned instead. The sequence is\n    started with value ``x0``.\n\n    Note: more than the first lambda + mu terms may be returned and this\n    is the cost of cycle detection with Brent's method; there are, however,\n    generally less terms calculated than would have been calculated if the\n    proper ending point were determined, e.g. by using Floyd's method.\n\n    >>> from sympy.ntheory.generate import cycle_length\n\n    This will yield successive values of i <-- func(i):\n\n        >>> def iter(func, i):\n        ...     while 1:\n        ...         ii = func(i)\n        ...         yield ii\n        ...         i = ii\n        ...\n\n    A function is defined:\n\n        >>> func = lambda i: (i**2 + 1) % 51\n\n    and given a seed of 4 and the mu and lambda terms calculated:\n\n        >>> next(cycle_length(func, 4))\n        (6, 2)\n\n    We can see what is meant by looking at the output:\n\n        >>> n = cycle_length(func, 4, values=True)\n        >>> list(ni for ni in n)\n        [17, 35, 2, 5, 26, 14, 44, 50, 2, 5, 26, 14]\n\n    There are 6 repeating values after the first 2.\n\n    If a sequence is suspected of being longer than you might wish, ``nmax``\n    can be used to exit early (and mu will be returned as None):\n\n        >>> next(cycle_length(func, 4, nmax = 4))\n        (4, None)\n        >>> [ni for ni in cycle_length(func, 4, nmax = 4, values=True)]\n        [17, 35, 2, 5]\n\n    Code modified from:\n        https://en.wikipedia.org/wiki/Cycle_detection.\n    "
    nmax = int(nmax or 0)
    power = lam = 1
    (tortoise, hare) = (x0, f(x0))
    i = 0
    while tortoise != hare and (not nmax or i < nmax):
        i += 1
        if power == lam:
            tortoise = hare
            power *= 2
            lam = 0
        if values:
            yield hare
        hare = f(hare)
        lam += 1
    if nmax and i == nmax:
        if values:
            return
        else:
            yield (nmax, None)
            return
    if not values:
        mu = 0
        tortoise = hare = x0
        for i in range(lam):
            hare = f(hare)
        while tortoise != hare:
            tortoise = f(tortoise)
            hare = f(hare)
            mu += 1
        if mu:
            mu -= 1
        yield (lam, mu)

def composite(nth):
    if False:
        print('Hello World!')
    ' Return the nth composite number, with the composite numbers indexed as\n        composite(1) = 4, composite(2) = 6, etc....\n\n        Examples\n        ========\n\n        >>> from sympy import composite\n        >>> composite(36)\n        52\n        >>> composite(1)\n        4\n        >>> composite(17737)\n        20000\n\n        See Also\n        ========\n\n        sympy.ntheory.primetest.isprime : Test if n is prime\n        primerange : Generate all primes in a given range\n        primepi : Return the number of primes less than or equal to n\n        prime : Return the nth prime\n        compositepi : Return the number of positive composite numbers less than or equal to n\n    '
    n = as_int(nth)
    if n < 1:
        raise ValueError('nth must be a positive integer; composite(1) == 4')
    composite_arr = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18]
    if n <= 10:
        return composite_arr[n - 1]
    (a, b) = (4, sieve._list[-1])
    if n <= b - primepi(b) - 1:
        while a < b - 1:
            mid = a + b >> 1
            if mid - primepi(mid) - 1 > n:
                b = mid
            else:
                a = mid
        if isprime(a):
            a -= 1
        return a
    from sympy.functions.elementary.exponential import log
    from sympy.functions.special.error_functions import li
    a = 4
    b = int(n * (log(n) + log(log(n))))
    while a < b:
        mid = a + b >> 1
        if mid - li(mid) - 1 > n:
            b = mid
        else:
            a = mid + 1
    n_composites = a - primepi(a) - 1
    while n_composites > n:
        if not isprime(a):
            n_composites -= 1
        a -= 1
    if isprime(a):
        a -= 1
    return a

def compositepi(n):
    if False:
        while True:
            i = 10
    ' Return the number of positive composite numbers less than or equal to n.\n        The first positive composite is 4, i.e. compositepi(4) = 1.\n\n        Examples\n        ========\n\n        >>> from sympy import compositepi\n        >>> compositepi(25)\n        15\n        >>> compositepi(1000)\n        831\n\n        See Also\n        ========\n\n        sympy.ntheory.primetest.isprime : Test if n is prime\n        primerange : Generate all primes in a given range\n        prime : Return the nth prime\n        primepi : Return the number of primes less than or equal to n\n        composite : Return the nth composite number\n    '
    n = int(n)
    if n < 4:
        return 0
    return n - primepi(n) - 1