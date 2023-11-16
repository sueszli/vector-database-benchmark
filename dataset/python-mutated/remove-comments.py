class Solution(object):

    def removeComments(self, source):
        if False:
            i = 10
            return i + 15
        '\n        :type source: List[str]\n        :rtype: List[str]\n        '
        in_block = False
        (result, newline) = ([], [])
        for line in source:
            i = 0
            while i < len(line):
                if not in_block and i + 1 < len(line) and (line[i:i + 2] == '/*'):
                    in_block = True
                    i += 1
                elif in_block and i + 1 < len(line) and (line[i:i + 2] == '*/'):
                    in_block = False
                    i += 1
                elif not in_block and i + 1 < len(line) and (line[i:i + 2] == '//'):
                    break
                elif not in_block:
                    newline.append(line[i])
                i += 1
            if newline and (not in_block):
                result.append(''.join(newline))
                newline = []
        return result