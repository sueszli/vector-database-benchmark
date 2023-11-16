def recursive_binomial_coefficient(n, k):
    if False:
        print('Hello World!')
    'Calculates the binomial coefficient, C(n,k), with n>=k using recursion\n    Time complexity is O(k), so can calculate fairly quickly for large values of k.\n\n    >>> recursive_binomial_coefficient(5,0)\n    1\n\n    >>> recursive_binomial_coefficient(8,2)\n    28\n\n    >>> recursive_binomial_coefficient(500,300)\n    5054949849935535817667719165973249533761635252733275327088189563256013971725761702359997954491403585396607971745777019273390505201262259748208640\n\n    '
    if k > n:
        raise ValueError('Invalid Inputs, ensure that n >= k')
    if k == 0 or n == k:
        return 1
    if k > n / 2:
        return recursive_binomial_coefficient(n, n - k)
    return int(n / k * recursive_binomial_coefficient(n - 1, k - 1))