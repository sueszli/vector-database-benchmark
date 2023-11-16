class ArrayReader(object):

    def query(self, a, b, c, d):
        if False:
            print('Hello World!')
        '\n        :type a, b, c, d: int\n        :rtype int\n        '
        pass

    def length(self):
        if False:
            i = 10
            return i + 15
        '\n        :rtype int\n        '
        pass

class Solution(object):

    def guessMajority(self, reader):
        if False:
            print('Hello World!')
        '\n        :type reader: ArrayReader\n        :rtype: integer\n        '
        (count_a, count_b, idx_b) = (1, 0, None)
        value_0_1_2_3 = reader.query(0, 1, 2, 3)
        for i in reversed(xrange(4, reader.length())):
            value_0_1_2_i = reader.query(0, 1, 2, i)
            if value_0_1_2_i == value_0_1_2_3:
                count_a = count_a + 1
            else:
                (count_b, idx_b) = (count_b + 1, i)
        value_0_1_2_4 = value_0_1_2_i
        for i in xrange(3):
            value_a_b_3_4 = reader.query(*[v for v in [0, 1, 2, 3, 4] if v != i])
            if value_a_b_3_4 == value_0_1_2_4:
                count_a = count_a + 1
            else:
                (count_b, idx_b) = (count_b + 1, i)
        if count_a == count_b:
            return -1
        return 3 if count_a > count_b else idx_b