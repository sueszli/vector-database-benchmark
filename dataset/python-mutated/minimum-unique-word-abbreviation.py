class Solution(object):

    def minAbbreviation(self, target, dictionary):
        if False:
            print('Hello World!')
        '\n        :type target: str\n        :type dictionary: List[str]\n        :rtype: str\n        '

        def bits_to_abbr_len(targets, bits):
            if False:
                i = 10
                return i + 15
            total = 0
            pre = 0
            for i in xrange(len(target)):
                if bits & 1:
                    if i - pre > 0:
                        total += len(str(i - pre))
                    pre = i + 1
                    total += 1
                elif i == len(target) - 1:
                    total += len(str(i - pre + 1))
                bits >>= 1
            return total

        def bits_to_abbr(targets, bits):
            if False:
                while True:
                    i = 10
            abbr = []
            pre = 0
            for i in xrange(len(target)):
                if bits & 1:
                    if i - pre > 0:
                        abbr.append(str(i - pre))
                    pre = i + 1
                    abbr.append(target[i])
                elif i == len(target) - 1:
                    abbr.append(str(i - pre + 1))
                bits >>= 1
            return ''.join(abbr)
        diffs = []
        for word in dictionary:
            if len(word) != len(target):
                continue
            diffs.append(sum((2 ** i for (i, c) in enumerate(word) if target[i] != c)))
        if not diffs:
            return str(len(target))
        result = 2 ** len(target) - 1
        for mask in xrange(2 ** len(target)):
            if all((d & mask for d in diffs)) and bits_to_abbr_len(target, mask) < bits_to_abbr_len(target, result):
                result = mask
        return bits_to_abbr(target, result)

class Solution2(object):

    def minAbbreviation(self, target, dictionary):
        if False:
            i = 10
            return i + 15
        '\n        :type target: str\n        :type dictionary: List[str]\n        :rtype: str\n        '

        def bits_to_abbr(targets, bits):
            if False:
                while True:
                    i = 10
            abbr = []
            pre = 0
            for i in xrange(len(target)):
                if bits & 1:
                    if i - pre > 0:
                        abbr.append(str(i - pre))
                    pre = i + 1
                    abbr.append(target[i])
                elif i == len(target) - 1:
                    abbr.append(str(i - pre + 1))
                bits >>= 1
            return ''.join(abbr)
        diffs = []
        for word in dictionary:
            if len(word) != len(target):
                continue
            diffs.append(sum((2 ** i for (i, c) in enumerate(word) if target[i] != c)))
        if not diffs:
            return str(len(target))
        result = target
        for mask in xrange(2 ** len(target)):
            abbr = bits_to_abbr(target, mask)
            if all((d & mask for d in diffs)) and len(abbr) < len(result):
                result = abbr
        return result