def countDigits(N):
    if False:
        print('Hello World!')
    if N == 0:
        return 1
    if N < 0:
        N = -N
    if N >= 10:
        N //= 10
        return countDigits(N) + 1
    else:
        return 1
print(countDigits(12688))
print(countDigits(375))
print(countDigits(64))
print(countDigits(-3459))