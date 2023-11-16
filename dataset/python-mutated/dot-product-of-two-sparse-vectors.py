class SparseVector:

    def __init__(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        '
        self.lookup = {i: v for (i, v) in enumerate(nums) if v}

    def dotProduct(self, vec):
        if False:
            return 10
        "\n        :type vec: 'SparseVector'\n        :rtype: int\n        "
        if len(self.lookup) > len(vec.lookup):
            (self, vec) = (vec, self)
        return sum((v * vec.lookup[i] for (i, v) in self.lookup.iteritems() if i in vec.lookup))