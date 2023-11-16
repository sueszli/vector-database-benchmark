class Solution(object):

    def findSmallestRegion(self, regions, region1, region2):
        if False:
            print('Hello World!')
        '\n        :type regions: List[List[str]]\n        :type region1: str\n        :type region2: str\n        :rtype: str\n        '
        parents = {region[i]: region[0] for region in regions for i in xrange(1, len(region))}
        lookup = {region1}
        while region1 in parents:
            region1 = parents[region1]
            lookup.add(region1)
        while region2 not in lookup:
            region2 = parents[region2]
        return region2