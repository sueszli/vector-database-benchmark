def palindrome_checker(sequence):
    if False:
        return 10

    def chars(sequence):
        if False:
            return 10
        sequence = sequence.lower()
        ans = ''
        for char in sequence:
            if char in 'abcdefghijklmnopqrstuvxwyz1234567890':
                ans = ans + char
        return ans

    def check(sequence):
        if False:
            for i in range(10):
                print('nop')
        if len(sequence) <= 1:
            return True
        else:
            return sequence[0] == sequence[-1] and check(sequence[1:-1])
    return check(chars(sequence))
print(palindrome_checker('2022/22/02'))
print(palindrome_checker("Madam, in Eden, I'm Adam"))