import itertools

class Solution(object):

    def distanceBetweenBusStops(self, distance, start, destination):
        if False:
            return 10
        '\n        :type distance: List[int]\n        :type start: int\n        :type destination: int\n        :rtype: int\n        '
        if start > destination:
            (start, destination) = (destination, start)
        s_to_d = sum(itertools.islice(distance, start, destination))
        d_to_s = sum(itertools.islice(distance, 0, start)) + sum(itertools.islice(distance, destination, len(distance)))
        return min(s_to_d, d_to_s)