def isHeterogram(string: str) -> bool:
    if False:
        i = 10
        return i + 15
    letters = set()
    for char in string:
        if char in letters:
            return False
        else:
            letters.add(char)
    return True

def isPangram(string: str) -> bool:
    if False:
        i = 10
        return i + 15
    alphabet = set('abcdefghijklmnopqrstuvwxyz')
    letters = set(filter(str.isalpha, string.lower()))
    return letters == alphabet
if __name__ == '__main__':
    string_test_1 = 'The quick brown fox jumps over the lazy dog'
    string_test_2 = 'murcielago'
    result = ''
    result += 'Es heterograma ' if isHeterogram(string_test_2) else 'No es heterograma '
    result += 'y es panagrama.' if isPangram(string_test_2) else 'y no es panagrama'
    print(result)