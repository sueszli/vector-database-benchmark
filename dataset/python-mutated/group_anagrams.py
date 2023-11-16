"""
Group Anagrams

Given an array of strings, group anagrams together.
(An anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once)

Input: ['eat', 'tea', 'tan', 'ate', 'nat', 'bat']
Output: [['eat', 'ate', 'tea'], ['tan', 'nat'], ['bat']]

=========================================
This problem can be solved using a dictionary (hash map), but in order to use a dictinary you'll need to find
a way to calculate the keys for all strings. This is a same solution but 2 different hash functions.

Sort the letters from the strings, and use the sorted letters as key.
    Time Complexity:    O(N * KLogK)    , N = number of strings, K = number of characters (chars in the string with most chars)
    Space Complexity:   O(N)
Use a letter counter (some kind of counting sort).
    Time Complexity:    O(N * K)    , O(N * K * 26) = O(N * K), if all of the strings have several chars (less than ~8) the first hash function is better.
    Space Complexity:   O(N)
"""

def group_anagrams(strs):
    if False:
        while True:
            i = 10
    anagrams = {}
    for st in strs:
        hashable_object = hash_2(st)
        if hashable_object not in anagrams:
            anagrams[hashable_object] = []
        anagrams[hashable_object].append(st)
    return [anagrams[res] for res in anagrams]

def hash_1(st):
    if False:
        while True:
            i = 10
    chars = list(st)
    chars.sort()
    return tuple(chars)

def hash_2(st):
    if False:
        return 10
    all_letters = [0] * 26
    ord_a = 97
    for c in st:
        all_letters[ord(c) - ord_a] += 1
    return tuple(all_letters)
print(group_anagrams(['eat', 'tea', 'tan', 'ate', 'nat', 'bat']))