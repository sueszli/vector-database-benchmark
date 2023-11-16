def find_nth_digit(n):
    if False:
        i = 10
        return i + 15
    'find the nth digit of given number.\n    1. find the length of the number where the nth digit is from.\n    2. find the actual number where the nth digit is from\n    3. find the nth digit and return\n    '
    length = 1
    count = 9
    start = 1
    while n > length * count:
        n -= length * count
        length += 1
        count *= 10
        start *= 10
    start += (n - 1) / length
    s = str(start)
    return int(s[(n - 1) % length])