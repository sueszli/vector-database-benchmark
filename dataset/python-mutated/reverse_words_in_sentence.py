"""
Reverse words in sentence

Reverse words in a given string, in linear time complexity.

Input: 'i like this program very much'
Output: 'much very program this like i'

Input: 'how are you'
Output: 'you are how'

=========================================
First, find each word and reverse it (in place, by swapping the letters),
after all words are reversed, reverse the whole sentence (in place, by swapping the letters)
and the first word will be last and will be in the original form.
In Python, the string manipulation operations are too slow (string is immutable), because of that we need to convert the string into array.
In C/C++, the Space complexity will be O(1) (because the strings are just arrays with chars).
    Time Complexity:    O(N)
    Space Complexity:   O(N)
"""

def reverse_words_in_sentence(sentence):
    if False:
        while True:
            i = 10
    arr = [c for c in sentence]
    n = len(arr)
    last_idx = n - 1
    start = 0
    for i in range(n):
        if arr[i] == ' ':
            reverse_array(arr, start, i - 1)
            start = i + 1
    reverse_array(arr, start, last_idx)
    reverse_array(arr, 0, last_idx)
    return ''.join(arr)

def reverse_array(arr, start, end):
    if False:
        i = 10
        return i + 15
    while start < end:
        (arr[start], arr[end]) = (arr[end], arr[start])
        start += 1
        end -= 1
print(reverse_words_in_sentence('i like this program very much'))
print(reverse_words_in_sentence('how are you'))