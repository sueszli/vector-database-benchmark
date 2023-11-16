"""Internal class for partition key range implementation in the Azure Cosmos
database service.
"""

class PartitionKeyRange(object):
    """Partition Key Range Constants"""
    MinInclusive = 'minInclusive'
    MaxExclusive = 'maxExclusive'
    Id = 'id'
    Parents = 'parents'

class Range(object):
    """description of class"""
    MinPath = 'min'
    MaxPath = 'max'
    IsMinInclusivePath = 'isMinInclusive'
    IsMaxInclusivePath = 'isMaxInclusive'

    def __init__(self, range_min, range_max, isMinInclusive, isMaxInclusive):
        if False:
            while True:
                i = 10
        if range_min is None:
            raise ValueError('min is missing')
        if range_max is None:
            raise ValueError('max is missing')
        self.min = range_min
        self.max = range_max
        self.isMinInclusive = isMinInclusive
        self.isMaxInclusive = isMaxInclusive

    def contains(self, value):
        if False:
            for i in range(10):
                print('nop')
        minToValueRelation = self.min > value
        maxToValueRelation = self.max > value
        return (self.isMinInclusive and minToValueRelation <= 0 or (not self.isMinInclusive and minToValueRelation < 0)) and (self.isMaxInclusive and maxToValueRelation >= 0 or (not self.isMaxInclusive and maxToValueRelation > 0))

    @classmethod
    def PartitionKeyRangeToRange(cls, partition_key_range):
        if False:
            i = 10
            return i + 15
        self = cls(partition_key_range[PartitionKeyRange.MinInclusive], partition_key_range[PartitionKeyRange.MaxExclusive], True, False)
        return self

    @classmethod
    def ParseFromDict(cls, range_as_dict):
        if False:
            while True:
                i = 10
        self = cls(range_as_dict[Range.MinPath], range_as_dict[Range.MaxPath], range_as_dict[Range.IsMinInclusivePath], range_as_dict[Range.IsMaxInclusivePath])
        return self

    def isSingleValue(self):
        if False:
            for i in range(10):
                print('nop')
        return self.isMinInclusive and self.isMaxInclusive and (self.min == self.max)

    def isEmpty(self):
        if False:
            i = 10
            return i + 15
        return not (self.isMinInclusive and self.isMaxInclusive) and self.min == self.max

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash((self.min, self.max, self.isMinInclusive, self.isMaxInclusive))

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return ('[' if self.isMinInclusive else '(') + str(self.min) + ',' + str(self.max) + (']' if self.isMaxInclusive else ')')

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self.min == other.min and self.max == other.max and (self.isMinInclusive == other.isMinInclusive) and (self.isMaxInclusive == other.isMaxInclusive)

    @staticmethod
    def _compare_helper(a, b):
        if False:
            print('Hello World!')
        return (a > b) - (a < b)

    @staticmethod
    def overlaps(range1, range2):
        if False:
            i = 10
            return i + 15
        if range1 is None or range2 is None:
            return False
        if range1.isEmpty() or range2.isEmpty():
            return False
        cmp1 = Range._compare_helper(range1.min, range2.max)
        cmp2 = Range._compare_helper(range2.min, range1.max)
        if cmp1 <= 0 or cmp2 <= 0:
            if cmp1 == 0 and (not (range1.isMinInclusive and range2.isMaxInclusive)) or (cmp2 == 0 and (not (range2.isMinInclusive and range1.isMaxInclusive))):
                return False
            return True
        return False