class InputType(object):
    INVALID = 0
    SPACE = 1
    SIGN = 2
    DIGIT = 3
    DOT = 4
    EXPONENT = 5

class Solution(object):

    def isNumber(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :rtype: bool\n        '
        transition_table = [[-1, 0, 3, 1, 2, -1], [-1, 8, -1, 1, 4, 5], [-1, -1, -1, 4, -1, -1], [-1, -1, -1, 1, 2, -1], [-1, 8, -1, 4, -1, 5], [-1, -1, 6, 7, -1, -1], [-1, -1, -1, 7, -1, -1], [-1, 8, -1, 7, -1, -1], [-1, 8, -1, -1, -1, -1]]
        state = 0
        for char in s:
            inputType = InputType.INVALID
            if char.isspace():
                inputType = InputType.SPACE
            elif char == '+' or char == '-':
                inputType = InputType.SIGN
            elif char.isdigit():
                inputType = InputType.DIGIT
            elif char == '.':
                inputType = InputType.DOT
            elif char == 'e' or char == 'E':
                inputType = InputType.EXPONENT
            state = transition_table[state][inputType]
            if state == -1:
                return False
        return state == 1 or state == 4 or state == 7 or (state == 8)

class Solution2(object):

    def isNumber(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: bool\n        '
        import re
        return bool(re.match('^\\s*[\\+-]?((\\d+(\\.\\d*)?)|\\.\\d+)([eE][\\+-]?\\d+)?\\s*$', s))