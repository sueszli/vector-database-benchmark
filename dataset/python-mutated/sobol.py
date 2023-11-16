"""
  Licensing:
    This code is distributed under the MIT license.

  Authors:
    Original FORTRAN77 version of i4_sobol by Bennett Fox.
    MATLAB version by John Burkardt.
    PYTHON version by Corrado Chisari

    Original Python version of is_prime by Corrado Chisari

    Original MATLAB versions of other functions by John Burkardt.
    PYTHON versions by Corrado Chisari

    Original code is available at
    http://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html

    Note: the i4 prefix means that the function takes a numeric argument or
          returns a number which is interpreted inside the function as a 4
          byte integer
    Note: the r4 prefix means that the function takes a numeric argument or
          returns a number which is interpreted inside the function as a 4
          byte float
"""
import math
import sys
import numpy as np
atmost = None
dim_max = None
dim_num_save = None
initialized = None
lastq = None
log_max = None
maxcol = None
poly = None
recipd = None
seed_save = None
v = None

def i4_bit_hi1(n):
    if False:
        while True:
            i = 10
    '\n     I4_BIT_HI1 returns the position of the high 1 bit base 2 in an I4.\n\n      Discussion:\n\n        An I4 is an integer ( kind = 4 ) value.\n\n      Example:\n\n           N    Binary    Hi 1\n        ----    --------  ----\n           0           0     0\n           1           1     1\n           2          10     2\n           3          11     2\n           4         100     3\n           5         101     3\n           6         110     3\n           7         111     3\n           8        1000     4\n           9        1001     4\n          10        1010     4\n          11        1011     4\n          12        1100     4\n          13        1101     4\n          14        1110     4\n          15        1111     4\n          16       10000     5\n          17       10001     5\n        1023  1111111111    10\n        1024 10000000000    11\n        1025 10000000001    11\n\n      Licensing:\n\n        This code is distributed under the GNU LGPL license.\n\n      Modified:\n\n        26 October 2014\n\n      Author:\n\n        John Burkardt\n\n      Parameters:\n\n        Input, integer N, the integer to be measured.\n        N should be nonnegative.  If N is nonpositive, the function\n        will always be 0.\n\n        Output, integer BIT, the position of the highest bit.\n\n    '
    i = n
    bit = 0
    while True:
        if i <= 0:
            break
        bit = bit + 1
        i = i // 2
    return bit

def i4_bit_lo0(n):
    if False:
        while True:
            i = 10
    '\n     I4_BIT_LO0 returns the position of the low 0 bit base 2 in an I4.\n\n      Discussion:\n\n        An I4 is an integer ( kind = 4 ) value.\n\n      Example:\n\n           N    Binary    Lo 0\n        ----    --------  ----\n           0           0     1\n           1           1     2\n           2          10     1\n           3          11     3\n           4         100     1\n           5         101     2\n           6         110     1\n           7         111     4\n           8        1000     1\n           9        1001     2\n          10        1010     1\n          11        1011     3\n          12        1100     1\n          13        1101     2\n          14        1110     1\n          15        1111     5\n          16       10000     1\n          17       10001     2\n        1023  1111111111    11\n        1024 10000000000     1\n        1025 10000000001     2\n\n      Licensing:\n\n        This code is distributed under the GNU LGPL license.\n\n      Modified:\n\n        08 February 2018\n\n      Author:\n\n        John Burkardt\n\n      Parameters:\n\n        Input, integer N, the integer to be measured.\n        N should be nonnegative.\n\n        Output, integer BIT, the position of the low 1 bit.\n\n    '
    bit = 0
    i = n
    while True:
        bit = bit + 1
        i2 = i // 2
        if i == 2 * i2:
            break
        i = i2
    return bit

def i4_sobol_generate(m, n, skip):
    if False:
        for i in range(10):
            print('nop')
    '\n\n\n     I4_SOBOL_GENERATE generates a Sobol dataset.\n\n      Licensing:\n\n        This code is distributed under the MIT license.\n\n      Modified:\n\n        22 February 2011\n\n      Author:\n\n        Original MATLAB version by John Burkardt.\n        PYTHON version by Corrado Chisari\n\n      Parameters:\n\n        Input, integer M, the spatial dimension.\n\n        Input, integer N, the number of points to generate.\n\n        Input, integer SKIP, the number of initial points to skip.\n\n        Output, real R(M,N), the points.\n\n    '
    r = np.zeros((m, n))
    for j in range(1, n + 1):
        seed = skip + j - 2
        [r[0:m, j - 1], seed] = i4_sobol(m, seed)
    return r

def i4_sobol(dim_num, seed):
    if False:
        return 10
    '\n\n\n     I4_SOBOL generates a new quasirandom Sobol vector with each call.\n\n      Discussion:\n\n        The routine adapts the ideas of Antonov and Saleev.\n\n      Licensing:\n\n        This code is distributed under the MIT license.\n\n      Modified:\n\n        22 February 2011\n\n      Author:\n\n        Original FORTRAN77 version by Bennett Fox.\n        MATLAB version by John Burkardt.\n        PYTHON version by Corrado Chisari\n\n      Reference:\n\n        Antonov, Saleev,\n        USSR Computational Mathematics and Mathematical Physics,\n        olume 19, 19, pages 252 - 256.\n\n        Paul Bratley, Bennett Fox,\n        Algorithm 659:\n        Implementing Sobol\'s Quasirandom Sequence Generator,\n        ACM Transactions on Mathematical Software,\n        Volume 14, Number 1, pages 88-100, 1988.\n\n        Bennett Fox,\n        Algorithm 647:\n        Implementation and Relative Efficiency of Quasirandom\n        Sequence Generators,\n        ACM Transactions on Mathematical Software,\n        Volume 12, Number 4, pages 362-376, 1986.\n\n        Ilya Sobol,\n        USSR Computational Mathematics and Mathematical Physics,\n        Volume 16, pages 236-242, 1977.\n\n        Ilya Sobol, Levitan,\n        The Production of Points Uniformly Distributed in a Multidimensional\n        Cube (in Russian),\n        Preprint IPM Akad. Nauk SSSR,\n        Number 40, Moscow 1976.\n\n      Parameters:\n\n        Input, integer DIM_NUM, the number of spatial dimensions.\n        DIM_NUM must satisfy 1 <= DIM_NUM <= 40.\n\n        Input/output, integer SEED, the "seed" for the sequence.\n        This is essentially the index in the sequence of the quasirandom\n        value to be generated.    On output, SEED has been set to the\n        appropriate next value, usually simply SEED+1.\n        If SEED is less than 0 on input, it is treated as though it were 0.\n        An input value of 0 requests the first (0-th) element of the sequence.\n\n        Output, real QUASI(DIM_NUM), the next quasirandom vector.\n\n    '
    global atmost
    global dim_max
    global dim_num_save
    global initialized
    global lastq
    global log_max
    global maxcol
    global poly
    global recipd
    global seed_save
    global v
    if not initialized or dim_num != dim_num_save:
        initialized = 1
        dim_max = 40
        dim_num_save = -1
        log_max = 30
        seed_save = -1
        v = np.zeros((dim_max, log_max))
        v[0:40, 0] = np.transpose([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        v[2:40, 1] = np.transpose([1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3])
        v[3:40, 2] = np.transpose([7, 5, 1, 3, 3, 7, 5, 5, 7, 7, 1, 3, 3, 7, 5, 1, 1, 5, 3, 3, 1, 7, 5, 1, 3, 3, 7, 5, 1, 1, 5, 7, 7, 5, 1, 3, 3])
        v[5:40, 3] = np.transpose([1, 7, 9, 13, 11, 1, 3, 7, 9, 5, 13, 13, 11, 3, 15, 5, 3, 15, 7, 9, 13, 9, 1, 11, 7, 5, 15, 1, 15, 11, 5, 3, 1, 7, 9])
        v[7:40, 4] = np.transpose([9, 3, 27, 15, 29, 21, 23, 19, 11, 25, 7, 13, 17, 1, 25, 29, 3, 31, 11, 5, 23, 27, 19, 21, 5, 1, 17, 13, 7, 15, 9, 31, 9])
        v[13:40, 5] = np.transpose([37, 33, 7, 5, 11, 39, 63, 27, 17, 15, 23, 29, 3, 21, 13, 31, 25, 9, 49, 33, 19, 29, 11, 19, 27, 15, 25])
        v[19:40, 6] = np.transpose([13, 33, 115, 41, 79, 17, 29, 119, 75, 73, 105, 7, 59, 65, 21, 3, 113, 61, 89, 45, 107])
        v[37:40, 7] = np.transpose([7, 23, 39])
        poly = [1, 3, 7, 11, 13, 19, 25, 37, 59, 47, 61, 55, 41, 67, 97, 91, 109, 103, 115, 131, 193, 137, 145, 143, 241, 157, 185, 167, 229, 171, 213, 191, 253, 203, 211, 239, 247, 285, 369, 299]
        atmost = 2 ** log_max - 1
        maxcol = i4_bit_hi1(atmost)
        v[0, 0:maxcol] = 1
    if dim_num != dim_num_save:
        if dim_num < 1 or dim_max < dim_num:
            print('I4_SOBOL - Fatal error!')
            print('    The spatial dimension DIM_NUM should satisfy:')
            print('        1 <= DIM_NUM <= %d' % dim_max)
            print('    But this input value is DIM_NUM = %d' % dim_num)
            return None
        dim_num_save = dim_num
        for i in range(2, dim_num + 1):
            j = poly[i - 1]
            m = 0
            while True:
                j = math.floor(j / 2.0)
                if j <= 0:
                    break
                m = m + 1
            j = poly[i - 1]
            includ = np.zeros(m)
            for k in range(m, 0, -1):
                j2 = math.floor(j / 2.0)
                includ[k - 1] = j != 2 * j2
                j = j2
            for j in range(m + 1, maxcol + 1):
                newv = v[i - 1, j - m - 1]
                l_var = 1
                for k in range(1, m + 1):
                    l_var = 2 * l_var
                    if includ[k - 1]:
                        newv = np.bitwise_xor(int(newv), int(l_var * v[i - 1, j - k - 1]))
                v[i - 1, j - 1] = newv
        l_var = 1
        for j in range(maxcol - 1, 0, -1):
            l_var = 2 * l_var
            v[0:dim_num, j - 1] = v[0:dim_num, j - 1] * l_var
        recipd = 1.0 / (2 * l_var)
        lastq = np.zeros(dim_num)
    seed = int(math.floor(seed))
    if seed < 0:
        seed = 0
    if seed == 0:
        l_var = 1
        lastq = np.zeros(dim_num)
    elif seed == seed_save + 1:
        l_var = i4_bit_lo0(seed)
    elif seed <= seed_save:
        seed_save = 0
        lastq = np.zeros(dim_num)
        for seed_temp in range(int(seed_save), int(seed)):
            l_var = i4_bit_lo0(seed_temp)
            for i in range(1, dim_num + 1):
                lastq[i - 1] = np.bitwise_xor(int(lastq[i - 1]), int(v[i - 1, l_var - 1]))
        l_var = i4_bit_lo0(seed)
    elif seed_save + 1 < seed:
        for seed_temp in range(int(seed_save + 1), int(seed)):
            l_var = i4_bit_lo0(seed_temp)
            for i in range(1, dim_num + 1):
                lastq[i - 1] = np.bitwise_xor(int(lastq[i - 1]), int(v[i - 1, l_var - 1]))
        l_var = i4_bit_lo0(seed)
    if maxcol < l_var:
        print('I4_SOBOL - Fatal error!')
        print('    Too many calls!')
        print('    MAXCOL = %d\n' % maxcol)
        print('    L =            %d\n' % l_var)
        return None
    quasi = np.zeros(dim_num)
    for i in range(1, dim_num + 1):
        quasi[i - 1] = lastq[i - 1] * recipd
        lastq[i - 1] = np.bitwise_xor(int(lastq[i - 1]), int(v[i - 1, l_var - 1]))
    seed_save = seed
    seed = seed + 1
    return [quasi, seed]

def i4_uniform_ab(a, b, seed):
    if False:
        print('Hello World!')
    "\n\n\n     I4_UNIFORM_AB returns a scaled pseudorandom I4.\n\n      Discussion:\n\n        The pseudorandom number will be scaled to be uniformly distributed\n        between A and B.\n\n      Licensing:\n\n        This code is distributed under the GNU LGPL license.\n\n      Modified:\n\n        05 April 2013\n\n      Author:\n\n        John Burkardt\n\n      Reference:\n\n        Paul Bratley, Bennett Fox, Linus Schrage,\n        A Guide to Simulation,\n        Second Edition,\n        Springer, 1987,\n        ISBN: 0387964673,\n        LC: QA76.9.C65.B73.\n\n        Bennett Fox,\n        Algorithm 647:\n        Implementation and Relative Efficiency of Quasirandom\n        Sequence Generators,\n        ACM Transactions on Mathematical Software,\n        Volume 12, Number 4, December 1986, pages 362-376.\n\n        Pierre L'Ecuyer,\n        Random Number Generation,\n        in Handbook of Simulation,\n        edited by Jerry Banks,\n        Wiley, 1998,\n        ISBN: 0471134031,\n        LC: T57.62.H37.\n\n        Peter Lewis, Allen Goodman, James Miller,\n        A Pseudo-Random Number Generator for the System/360,\n        IBM Systems Journal,\n        Volume 8, Number 2, 1969, pages 136-143.\n\n      Parameters:\n\n        Input, integer A, B, the minimum and maximum acceptable values.\n\n        Input, integer SEED, a seed for the random number generator.\n\n        Output, integer C, the randomly chosen integer.\n\n        Output, integer SEED, the updated seed.\n\n    "
    i4_huge = 2147483647
    seed = int(seed)
    seed = seed % i4_huge
    if seed < 0:
        seed = seed + i4_huge
    if seed == 0:
        print('')
        print('I4_UNIFORM_AB - Fatal error!')
        print('  Input SEED = 0!')
        sys.exit('I4_UNIFORM_AB - Fatal error!')
    k = seed // 127773
    seed = 167 * (seed - k * 127773) - k * 2836
    if seed < 0:
        seed = seed + i4_huge
    r = seed * 4.656612875e-10
    a = round(a)
    b = round(b)
    r = (1.0 - r) * (min(a, b) - 0.5) + r * (max(a, b) + 0.5)
    value = round(r)
    value = max(value, min(a, b))
    value = min(value, max(a, b))
    value = int(value)
    return (value, seed)

def prime_ge(n):
    if False:
        while True:
            i = 10
    '\n\n\n     PRIME_GE returns the smallest prime greater than or equal to N.\n\n      Example:\n\n          N    PRIME_GE\n\n        -10     2\n          1     2\n          2     2\n          3     3\n          4     5\n          5     5\n          6     7\n          7     7\n          8    11\n          9    11\n         10    11\n\n      Licensing:\n\n        This code is distributed under the MIT license.\n\n      Modified:\n\n        22 February 2011\n\n      Author:\n\n        Original MATLAB version by John Burkardt.\n        PYTHON version by Corrado Chisari\n\n      Parameters:\n\n        Input, integer N, the number to be bounded.\n\n        Output, integer P, the smallest prime number that is greater\n        than or equal to N.\n\n    '
    p = max(math.ceil(n), 2)
    while not isprime(p):
        p = p + 1
    return p

def isprime(n):
    if False:
        return 10
    '\n\n\n     IS_PRIME returns True if N is a prime number, False otherwise\n\n      Licensing:\n\n        This code is distributed under the MIT license.\n\n      Modified:\n\n        22 February 2011\n\n      Author:\n\n        Corrado Chisari\n\n      Parameters:\n\n        Input, integer N, the number to be checked.\n\n        Output, boolean value, True or False\n\n    '
    if n != int(n) or n < 1:
        return False
    p = 2
    while p < n:
        if n % p == 0:
            return False
        p += 1
    return True

def r4_uniform_01(seed):
    if False:
        while True:
            i = 10
    '\n\n\n     R4_UNIFORM_01 returns a unit pseudorandom R4.\n\n      Discussion:\n\n        This routine implements the recursion\n\n          seed = 167 * seed mod ( 2^31 - 1 )\n          r = seed / ( 2^31 - 1 )\n\n        The integer arithmetic never requires more than 32 bits,\n        including a sign bit.\n\n        If the initial seed is 12345, then the first three computations are\n\n          Input     Output      R4_UNIFORM_01\n          SEED      SEED\n\n             12345   207482415  0.096616\n         207482415  1790989824  0.833995\n        1790989824  2035175616  0.947702\n\n      Licensing:\n\n        This code is distributed under the GNU LGPL license.\n\n      Modified:\n\n        04 April 2013\n\n      Author:\n\n        John Burkardt\n\n      Reference:\n\n        Paul Bratley, Bennett Fox, Linus Schrage,\n        A Guide to Simulation,\n        Second Edition,\n        Springer, 1987,\n        ISBN: 0387964673,\n        LC: QA76.9.C65.B73.\n\n        Bennett Fox,\n        Algorithm 647:\n        Implementation and Relative Efficiency of Quasirandom\n        Sequence Generators,\n        ACM Transactions on Mathematical Software,\n        Volume 12, Number 4, December 1986, pages 362-376.\n\n        Pierre L\'Ecuyer,\n        Random Number Generation,\n        in Handbook of Simulation,\n        edited by Jerry Banks,\n        Wiley, 1998,\n        ISBN: 0471134031,\n        LC: T57.62.H37.\n\n        Peter Lewis, Allen Goodman, James Miller,\n        A Pseudo-Random Number Generator for the System/360,\n        IBM Systems Journal,\n        Volume 8, Number 2, 1969, pages 136-143.\n\n      Parameters:\n\n        Input, integer SEED, the integer "seed" used to generate\n        the output random number.  SEED should not be 0.\n\n        Output, real R, a random value between 0 and 1.\n\n        Output, integer SEED, the updated seed.  This would\n        normally be used as the input seed on the next call.\n\n    '
    i4_huge = 2147483647
    if seed == 0:
        print('')
        print('R4_UNIFORM_01 - Fatal error!')
        print('  Input SEED = 0!')
        sys.exit('R4_UNIFORM_01 - Fatal error!')
    seed = seed % i4_huge
    if seed < 0:
        seed = seed + i4_huge
    k = seed // 127773
    seed = 167 * (seed - k * 127773) - k * 2836
    if seed < 0:
        seed = seed + i4_huge
    r = seed * 4.656612875e-10
    return (r, seed)

def r8mat_write(filename, m, n, a):
    if False:
        return 10
    '\n\n\n     R8MAT_WRITE writes an R8MAT to a file.\n\n      Licensing:\n\n        This code is distributed under the GNU LGPL license.\n\n      Modified:\n\n        12 October 2014\n\n      Author:\n\n        John Burkardt\n\n      Parameters:\n\n        Input, string FILENAME, the name of the output file.\n\n        Input, integer M, the number of rows in A.\n\n        Input, integer N, the number of columns in A.\n\n        Input, real A(M,N), the matrix.\n    '
    with open(filename, 'w') as output:
        for i in range(0, m):
            for j in range(0, n):
                s = '  %g' % a[i, j]
                output.write(s)
            output.write('\n')

def tau_sobol(dim_num):
    if False:
        for i in range(10):
            print('nop')
    '\n\n\n     TAU_SOBOL defines favorable starting seeds for Sobol sequences.\n\n      Discussion:\n\n        For spatial dimensions 1 through 13, this routine returns\n        a "favorable" value TAU by which an appropriate starting point\n        in the Sobol sequence can be determined.\n\n        These starting points have the form N = 2**K, where\n        for integration problems, it is desirable that\n                TAU + DIM_NUM - 1 <= K\n        while for optimization problems, it is desirable that\n                TAU < K.\n\n      Licensing:\n\n        This code is distributed under the MIT license.\n\n      Modified:\n\n        22 February 2011\n\n      Author:\n\n        Original FORTRAN77 version by Bennett Fox.\n        MATLAB version by John Burkardt.\n        PYTHON version by Corrado Chisari\n\n      Reference:\n\n        IA Antonov, VM Saleev,\n        USSR Computational Mathematics and Mathematical Physics,\n        Volume 19, 19, pages 252 - 256.\n\n        Paul Bratley, Bennett Fox,\n        Algorithm 659:\n        Implementing Sobol\'s Quasirandom Sequence Generator,\n        ACM Transactions on Mathematical Software,\n        Volume 14, Number 1, pages 88-100, 1988.\n\n        Bennett Fox,\n        Algorithm 647:\n        Implementation and Relative Efficiency of Quasirandom\n        Sequence Generators,\n        ACM Transactions on Mathematical Software,\n        Volume 12, Number 4, pages 362-376, 1986.\n\n        Stephen Joe, Frances Kuo\n        Remark on Algorithm 659:\n        Implementing Sobol\'s Quasirandom Sequence Generator,\n        ACM Transactions on Mathematical Software,\n        Volume 29, Number 1, pages 49-57, March 2003.\n\n        Ilya Sobol,\n        USSR Computational Mathematics and Mathematical Physics,\n        Volume 16, pages 236-242, 1977.\n\n        Ilya Sobol, YL Levitan,\n        The Production of Points Uniformly Distributed in a Multidimensional\n        Cube (in Russian),\n        Preprint IPM Akad. Nauk SSSR,\n        Number 40, Moscow 1976.\n\n      Parameters:\n\n                Input, integer DIM_NUM, the spatial dimension.    Only values\n                of 1 through 13 will result in useful responses.\n\n                Output, integer TAU, the value TAU.\n\n    '
    dim_max = 13
    tau_table = [0, 0, 1, 3, 5, 8, 11, 15, 19, 23, 27, 31, 35]
    if 1 <= dim_num <= dim_max:
        tau = tau_table[dim_num]
    else:
        tau = -1
    return tau