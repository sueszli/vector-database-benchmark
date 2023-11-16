import collections

class Solution(object):

    def findMinStep(self, board, hand):
        if False:
            return 10
        '\n        :type board: str\n        :type hand: str\n        :rtype: int\n        '

        def shrink(s):
            if False:
                print('Hello World!')
            while True:
                i = 0
                for start in xrange(len(s)):
                    while i < len(s) and s[start] == s[i]:
                        i += 1
                    if i - start >= 3:
                        s = s[0:start] + s[i:]
                        break
                else:
                    break
            return s

        def findMinStepHelper(board, hand, lookup):
            if False:
                print('Hello World!')
            if not board:
                return 0
            if not hand:
                return float('inf')
            if tuple(hand) in lookup[tuple(board)]:
                return lookup[tuple(board)][tuple(hand)]
            result = float('inf')
            for i in xrange(len(hand)):
                for j in xrange(len(board) + 1):
                    next_board = shrink(board[0:j] + hand[i:i + 1] + board[j:])
                    next_hand = hand[0:i] + hand[i + 1:]
                    result = min(result, findMinStepHelper(next_board, next_hand, lookup) + 1)
            lookup[tuple(board)][tuple(hand)] = result
            return result
        lookup = collections.defaultdict(dict)
        (board, hand) = (list(board), list(hand))
        result = findMinStepHelper(board, hand, lookup)
        return -1 if result == float('inf') else result
import collections

class Solution_TLE(object):

    def findMinStep(self, board, hand):
        if False:
            while True:
                i = 10
        '\n        :type board: str\n        :type hand: str\n        :rtype: int\n        '

        def shrink(s):
            if False:
                while True:
                    i = 10
            stack = []
            start = 0
            for i in xrange(len(s) + 1):
                if i == len(s) or s[i] != s[start]:
                    if stack and stack[-1][0] == s[start]:
                        stack[-1][1] += i - start
                        if stack[-1][1] >= 3:
                            stack.pop()
                    elif s and i - start < 3:
                        stack += ([s[start], i - start],)
                    start = i
            result = []
            for p in stack:
                result += [p[0]] * p[1]
            return result

        def findMinStepHelper(board, hand, lookup):
            if False:
                i = 10
                return i + 15
            if not board:
                return 0
            if not hand:
                return float('inf')
            if tuple(hand) in lookup[tuple(board)]:
                return lookup[tuple(board)][tuple(hand)]
            result = float('inf')
            for i in xrange(len(hand)):
                for j in xrange(len(board) + 1):
                    next_board = shrink(board[0:j] + hand[i:i + 1] + board[j:])
                    next_hand = hand[0:i] + hand[i + 1:]
                    result = min(result, findMinStepHelper(next_board, next_hand, lookup) + 1)
            lookup[tuple(board)][tuple(hand)] = result
            return result
        lookup = collections.defaultdict(dict)
        (board, hand) = (list(board), list(hand))
        result = findMinStepHelper(board, hand, lookup)
        return -1 if result == float('inf') else result
import collections

class Solution_GREEDY_ACCEPT_BUT_NOT_PROVED(object):

    def findMinStep(self, board, hand):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type board: str\n        :type hand: str\n        :rtype: int\n        '

        def shrink(s):
            if False:
                while True:
                    i = 10
            stack = []
            start = 0
            for i in xrange(len(s) + 1):
                if i == len(s) or s[i] != s[start]:
                    if stack and stack[-1][0] == s[start]:
                        stack[-1][1] += i - start
                        if stack[-1][1] >= 3:
                            stack.pop()
                    elif s and i - start < 3:
                        stack += ([s[start], i - start],)
                    start = i
            result = []
            for p in stack:
                result += [p[0]] * p[1]
            return result

        def findMinStepHelper2(board, hand, lookup):
            if False:
                print('Hello World!')
            result = float('inf')
            for i in xrange(len(hand)):
                for j in xrange(len(board) + 1):
                    next_board = shrink(board[0:j] + hand[i:i + 1] + board[j:])
                    next_hand = hand[0:i] + hand[i + 1:]
                    result = min(result, findMinStepHelper(next_board, next_hand, lookup) + 1)
            return result

        def find(board, c, j):
            if False:
                print('Hello World!')
            for i in xrange(j, len(board)):
                if board[i] == c:
                    return i
            return -1

        def findMinStepHelper(board, hand, lookup):
            if False:
                i = 10
                return i + 15
            if not board:
                return 0
            if not hand:
                return float('inf')
            if tuple(hand) in lookup[tuple(board)]:
                return lookup[tuple(board)][tuple(hand)]
            result = float('inf')
            for i in xrange(len(hand)):
                j = 0
                while j < len(board):
                    k = find(board, hand[i], j)
                    if k == -1:
                        break
                    if k < len(board) - 1 and board[k] == board[k + 1]:
                        next_board = shrink(board[0:k] + board[k + 2:])
                        next_hand = hand[0:i] + hand[i + 1:]
                        result = min(result, findMinStepHelper(next_board, next_hand, lookup) + 1)
                        k += 1
                    elif i > 0 and hand[i] == hand[i - 1]:
                        next_board = shrink(board[0:k] + board[k + 1:])
                        next_hand = hand[0:i - 1] + hand[i + 1:]
                        result = min(result, findMinStepHelper(next_board, next_hand, lookup) + 2)
                    j = k + 1
            lookup[tuple(board)][tuple(hand)] = result
            return result
        (board, hand) = (list(board), list(hand))
        hand.sort()
        result = findMinStepHelper(board, hand, collections.defaultdict(dict))
        if result == float('inf'):
            result = findMinStepHelper2(board, hand, collections.defaultdict(dict))
        return -1 if result == float('inf') else result

class Solution_WRONG_GREEDY_AND_NOT_ACCEPT_NOW(object):

    def findMinStep(self, board, hand):
        if False:
            print('Hello World!')
        '\n        :type board: str\n        :type hand: str\n        :rtype: int\n        '

        def shrink(s):
            if False:
                return 10
            stack = []
            start = 0
            for i in xrange(len(s) + 1):
                if i == len(s) or s[i] != s[start]:
                    if stack and stack[-1][0] == s[start]:
                        stack[-1][1] += i - start
                        if stack[-1][1] >= 3:
                            stack.pop()
                    elif s and i - start < 3:
                        stack += ([s[start], i - start],)
                    start = i
            result = []
            for p in stack:
                result += [p[0]] * p[1]
            return result

        def find(board, c, j):
            if False:
                print('Hello World!')
            for i in xrange(j, len(board)):
                if board[i] == c:
                    return i
            return -1

        def findMinStepHelper(board, hand, lookup):
            if False:
                while True:
                    i = 10
            if not board:
                return 0
            if not hand:
                return float('inf')
            if tuple(hand) in lookup[tuple(board)]:
                return lookup[tuple(board)][tuple(hand)]
            result = float('inf')
            for i in xrange(len(hand)):
                j = 0
                while j < len(board):
                    k = find(board, hand[i], j)
                    if k == -1:
                        break
                    if k < len(board) - 1 and board[k] == board[k + 1]:
                        next_board = shrink(board[0:k] + board[k + 2:])
                        next_hand = hand[0:i] + hand[i + 1:]
                        result = min(result, findMinStepHelper(next_board, next_hand, lookup) + 1)
                        k += 1
                    elif i > 0 and hand[i] == hand[i - 1]:
                        next_board = shrink(board[0:k] + board[k + 1:])
                        next_hand = hand[0:i - 1] + hand[i + 1:]
                        result = min(result, findMinStepHelper(next_board, next_hand, lookup) + 2)
                    j = k + 1
            lookup[tuple(board)][tuple(hand)] = result
            return result
        lookup = collections.defaultdict(dict)
        (board, hand) = (list(board), list(hand))
        hand.sort()
        result = findMinStepHelper(board, hand, lookup)
        return -1 if result == float('inf') else result