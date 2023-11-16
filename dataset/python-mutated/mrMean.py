"""
Created on Feb 28, 2011

@author: Peter
"""
from mrjob.job import MRJob

class MRmean(MRJob):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(MRmean, self).__init__(*args, **kwargs)
        self.inCount = 0
        self.inSum = 0
        self.inSqSum = 0

    def map(self, key, val):
        if False:
            return 10
        if False:
            yield
        inVal = float(val)
        self.inCount += 1
        self.inSum += inVal
        self.inSqSum += inVal * inVal

    def map_final(self):
        if False:
            for i in range(10):
                print('nop')
        mn = self.inSum / self.inCount
        mnSq = self.inSqSum / self.inCount
        yield (1, [self.inCount, mn, mnSq])

    def reduce(self, key, packedValues):
        if False:
            return 10
        cumVal = 0.0
        cumSumSq = 0.0
        cumN = 0.0
        for valArr in packedValues:
            nj = float(valArr[0])
            cumN += nj
            cumVal += nj * float(valArr[1])
            cumSumSq += nj * float(valArr[2])
        mean = cumVal / cumN
        var = (cumSumSq - 2 * mean * cumVal + cumN * mean * mean) / cumN
        yield (mean, var)

    def steps(self):
        if False:
            return 10
        return [self.mr(mapper=self.map, mapper_final=self.map_final, reducer=self.reduce)]
if __name__ == '__main__':
    MRmean.run()