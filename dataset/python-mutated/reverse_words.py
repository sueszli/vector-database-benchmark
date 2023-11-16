def reverse(array, i, j):
    if False:
        i = 10
        return i + 15
    while i < j:
        (array[i], array[j]) = (array[j], array[i])
        i += 1
        j -= 1

def reverse_words(string):
    if False:
        i = 10
        return i + 15
    arr = string.strip().split()
    n = len(arr)
    reverse(arr, 0, n - 1)
    return ' '.join(arr)
if __name__ == '__main__':
    test = 'I am keon kim and I like pizza'
    print(test)
    print(reverse_words(test))