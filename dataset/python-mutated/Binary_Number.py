def toBinary(n):
    if False:
        return 10
    if n <= 1:
        return str(n)
    bin_n = toBinary(n // 2)
    bin_n += str(n % 2)
    return ''.join(map(str, bin_n))
print(toBinary(10))
print(toBinary(5))
print(toBinary(20))