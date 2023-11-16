def myPow(x: float, n: int):
    if False:
        print('Hello World!')
    if n == 0:
        return 1
    elif x == 0:
        return 0
    elif n < 0:
        return 1 / myPow(x, -n)
    else:
        temp = myPow(x, n // 2)
        if n % 2 == 0:
            return temp * temp
        else:
            return temp * temp * x
print(myPow(2.0, 10))
print(myPow(2.1, 3))
print(myPow(2.0, -2))