"""
Find pairs with sum K

Given an array, find all pairs which sum is equal to K.

Input: [1, 2, 3, 4, 5, 5, 6, 7, 8, 9], 5
Output: [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]

=========================================
Save numbers as complements in a hashset and for each number search for the pair complement (K-number).
    Time Complexity:    O(N)
    Space Complexity:   O(N)
"""

def find_pairs(arr, K):
    if False:
        while True:
            i = 10
    complements = set()
    pair_complements = set()
    for el in arr:
        c = K - el
        if c in complements:
            pair_complements.add(c)
        complements.add(el)
    pairs = []
    for c in pair_complements:
        pairs.append((c, K - c))
    return pairs
print(find_pairs([1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 8, 8, 9], 10))