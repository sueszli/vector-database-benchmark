"""
Numbers can be regarded as product of its factors. For example,
8 = 2 x 2 x 2;
  = 2 x 4.


Write a function that takes an integer n and return all possible combinations
of its factors.Numbers can be regarded as product of its factors. For example,
8 = 2 x 2 x 2;
  = 2 x 4.

Examples:
input: 1
output:
[]


input: 37
output:
[]

input: 32
output:
[
  [2, 16],
  [2, 2, 8],
  [2, 2, 2, 4],
  [2, 2, 2, 2, 2],
"""

def get_factors(n):
    if False:
        while True:
            i = 10
    '[summary]\n    \n    Arguments:\n        n {[int]} -- [to analysed number]\n    \n    Returns:\n        [list of lists] -- [all factors of the number n]\n    '

    def factor(n, i, combi, res):
        if False:
            i = 10
            return i + 15
        '[summary]\n        helper function\n\n        Arguments:\n            n {[int]} -- [number]\n            i {[int]} -- [to tested divisor]\n            combi {[list]} -- [catch divisors]\n            res {[list]} -- [all factors of the number n]\n        \n        Returns:\n            [list] -- [res]\n        '
        while i * i <= n:
            if n % i == 0:
                res += (combi + [i, int(n / i)],)
                factor(n / i, i, combi + [i], res)
            i += 1
        return res
    return factor(n, 2, [], [])

def get_factors_iterative1(n):
    if False:
        i = 10
        return i + 15
    '[summary]\n    Computes all factors of n.\n    Translated the function get_factors(...) in\n    a call-stack modell.\n\n    Arguments:\n        n {[int]} -- [to analysed number]\n    \n    Returns:\n        [list of lists] -- [all factors]\n    '
    (todo, res) = ([(n, 2, [])], [])
    while todo:
        (n, i, combi) = todo.pop()
        while i * i <= n:
            if n % i == 0:
                res += (combi + [i, n // i],)
                (todo.append((n // i, i, combi + [i])),)
            i += 1
    return res

def get_factors_iterative2(n):
    if False:
        while True:
            i = 10
    '[summary]\n    analog as above\n\n    Arguments:\n        n {[int]} -- [description]\n    \n    Returns:\n        [list of lists] -- [all factors of n]\n    '
    (ans, stack, x) = ([], [], 2)
    while True:
        if x > n // x:
            if not stack:
                return ans
            ans.append(stack + [n])
            x = stack.pop()
            n *= x
            x += 1
        elif n % x == 0:
            stack.append(x)
            n //= x
        else:
            x += 1