class Solution(object):

    def addOperators(self, num, target):
        if False:
            return 10
        '\n        :type num: str\n        :type target: int\n        :rtype: List[str]\n        '
        (result, expr) = ([], [])
        (val, i) = (0, 0)
        val_str = ''
        while i < len(num):
            val = val * 10 + ord(num[i]) - ord('0')
            val_str += num[i]
            if str(val) != val_str:
                break
            expr.append(val_str)
            self.addOperatorsDFS(num, target, i + 1, 0, val, expr, result)
            expr.pop()
            i += 1
        return result

    def addOperatorsDFS(self, num, target, pos, operand1, operand2, expr, result):
        if False:
            for i in range(10):
                print('nop')
        if pos == len(num) and operand1 + operand2 == target:
            result.append(''.join(expr))
        else:
            (val, i) = (0, pos)
            val_str = ''
            while i < len(num):
                val = val * 10 + ord(num[i]) - ord('0')
                val_str += num[i]
                if str(val) != val_str:
                    break
                expr.append('+' + val_str)
                self.addOperatorsDFS(num, target, i + 1, operand1 + operand2, val, expr, result)
                expr.pop()
                expr.append('-' + val_str)
                self.addOperatorsDFS(num, target, i + 1, operand1 + operand2, -val, expr, result)
                expr.pop()
                expr.append('*' + val_str)
                self.addOperatorsDFS(num, target, i + 1, operand1, operand2 * val, expr, result)
                expr.pop()
                i += 1