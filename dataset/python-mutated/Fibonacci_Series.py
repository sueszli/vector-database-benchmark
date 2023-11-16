def fib(n):
    if False:
        print('Hello World!')
    if n == 0 or n == 1:
        return n
    else:
        return fib(n - 2) + fib(n - 1)
for i in range(6):
    print(fib(i), end=' ')