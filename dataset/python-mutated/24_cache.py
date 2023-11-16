from functools import wraps

def memoize(function):
    if False:
        i = 10
        return i + 15
    memo = {}

    @wraps(function)
    def wrapper(*args):
        if False:
            while True:
                i = 10
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper

@memoize
def fibonacci(n):
    if False:
        return 10
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def main():
    if False:
        while True:
            i = 10
    fibonacci(25)
if __name__ == '__main__':
    main()