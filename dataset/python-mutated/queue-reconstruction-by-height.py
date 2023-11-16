class Solution(object):

    def reconstructQueue(self, people):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type people: List[List[int]]\n        :rtype: List[List[int]]\n        '
        people.sort(key=lambda h_k: (-h_k[0], h_k[1]))
        blocks = [[]]
        for p in people:
            index = p[1]
            for (i, block) in enumerate(blocks):
                if index <= len(block):
                    break
                index -= len(block)
            block.insert(index, p)
            if len(block) * len(block) > len(people):
                blocks.insert(i + 1, block[len(block) / 2:])
                del block[len(block) / 2:]
        return [p for block in blocks for p in block]

class Solution2(object):

    def reconstructQueue(self, people):
        if False:
            while True:
                i = 10
        '\n        :type people: List[List[int]]\n        :rtype: List[List[int]]\n        '
        people.sort(key=lambda h_k1: (-h_k1[0], h_k1[1]))
        result = []
        for p in people:
            result.insert(p[1], p)
        return result