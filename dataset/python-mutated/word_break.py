"""
Word Break (Find the original words)

Given a dictionary of words and a string made up of those words (no spaces), return the original sentence in a list.
If there is more than one possible reconstruction, return solution with less words.
If there is no possible reconstruction, then return null.

Input: sentence = 'thequickbrownfox', words = ['quick', 'brown', 'the', 'fox']
Output: ['the', 'quick', 'brown', 'fox']

Input: sentence = 'bedbathandbeyond', words = ['bed', 'bath', 'bedbath', 'and', 'beyond']
Output: ['bedbath', 'and', 'beyond'] (['bed', 'bath', 'and', 'beyond] has more words)

=========================================
Optimized dynamic programming solution (more simpler solutions can be found here https://www.geeksforgeeks.org/word-break-problem-dp-32/)
    Time Complexity:    O(N*M)  , N = number of chars in the sentence, M = max word length
    Space Complexity:   O(N+W)  , W = number of words
Bonus solution: Backtracking, iterate the sentence construct a substring and check if that substring exist in the set of words.
If the end is reached but the last word doesn't exist in the words, go back 1 word from the result (backtracking).
* But this solution doesn't give the result with the smallest number of words (gives the first found result)
    Time Complexity:    O(?)    , (worst case, about O(W! * N), for example sentence='aaaaaac', words=['a','aa','aaa','aaaa','aaaaa', 'aaaaaa'])
    Space Complexity:   O(W)
"""
import math

def word_break(sentence, words):
    if False:
        return 10
    (n, w) = (len(sentence), len(words))
    if n == 0 or w == 0:
        return None
    dw = [-1 for i in range(n + 1)]
    dp = [math.inf for i in range(n + 1)]
    dp[0] = 0
    matched_indices = [0]
    dic = {}
    max_word = 0
    for i in range(w):
        dic[words[i]] = i
        max_word = max(max_word, len(words[i]))
    for i in range(1, n + 1):
        matched = False
        for j in range(len(matched_indices) - 1, -1, -1):
            matched_index = matched_indices[j]
            if matched_index < i - max_word:
                break
            subsentence = sentence[matched_index:i]
            if subsentence in dic and dp[matched_index] + 1 < dp[i]:
                dp[i] = dp[matched_index] + 1
                dw[i] = dic[subsentence]
                matched = True
        if matched:
            matched_indices.append(i)
    if dp[n] == math.inf:
        return None
    result = ['' for i in range(dp[n])]
    i = n
    j = dp[n] - 1
    while i > 0:
        result[j] = words[dw[i]]
        i -= len(words[dw[i]])
        j -= 1
    return result
from collections import deque

def word_break_backtracking(sentence, words):
    if False:
        i = 10
        return i + 15
    all_words = set()
    for i in range(len(words)):
        all_words.add(words[i])
    n = len(sentence)
    i = 0
    subsentence = ''
    result = deque()
    while i < n or len(subsentence) != 0:
        if i == n:
            i -= len(subsentence)
            if len(result) == 0:
                return None
            subsentence = result[-1]
            result.pop()
        subsentence += sentence[i]
        i += 1
        if subsentence in all_words:
            result.append(subsentence)
            subsentence = ''
    return list(result)
print(word_break('thequickbrownfox', ['quick', 'brown', 'the', 'fox']))
print(word_break('bedbathandbeyond', ['bed', 'bath', 'bedbath', 'and', 'beyond']))
print(word_break('bedbathandbeyond', ['bed', 'and', 'bath', 'bedbath', 'bathand', 'beyond', 'andbeyond']))
print(word_break('bedbathandbeyo', ['bed', 'bath', 'bedbath', 'bathand', 'beyond']))
print(word_break('3141592653589793238462643383279', ['314', '49', '9001', '15926535897', '14', '9323', '8462643383279', '4', '793']))
print(word_break('ilikelikeimangoiii', ['mobile', 'samsung', 'sam', 'sung', 'man', 'mango', 'icecream', 'and', 'go', 'i', 'like', 'ice', 'cream']))