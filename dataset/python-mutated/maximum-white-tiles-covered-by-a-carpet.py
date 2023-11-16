class Solution(object):

    def maximumWhiteTiles(self, tiles, carpetLen):
        if False:
            i = 10
            return i + 15
        '\n        :type tiles: List[List[int]]\n        :type carpetLen: int\n        :rtype: int\n        '
        tiles.sort()
        result = right = gap = 0
        for (left, (l, _)) in enumerate(tiles):
            if left - 1 >= 0:
                gap -= tiles[left][0] - tiles[left - 1][1] - 1
            r = l + carpetLen - 1
            while right + 1 < len(tiles) and r + 1 >= tiles[right + 1][0]:
                right += 1
                gap += tiles[right][0] - tiles[right - 1][1] - 1
            result = max(result, min(tiles[right][1] - tiles[left][0] + 1, carpetLen) - gap)
        return result

class Solution2(object):

    def maximumWhiteTiles(self, tiles, carpetLen):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type tiles: List[List[int]]\n        :type carpetLen: int\n        :rtype: int\n        '
        tiles.sort()
        result = left = gap = 0
        for right in xrange(len(tiles)):
            if right - 1 >= 0:
                gap += tiles[right][0] - tiles[right - 1][1] - 1
            l = tiles[right][1] - carpetLen + 1
            while not tiles[left][1] + 1 >= l:
                left += 1
                gap -= tiles[left][0] - tiles[left - 1][1] - 1
            result = max(result, min(tiles[right][1] - tiles[left][0] + 1, carpetLen) - gap)
        return result
import bisect

class Solution3(object):

    def maximumWhiteTiles(self, tiles, carpetLen):
        if False:
            return 10
        '\n        :type tiles: List[List[int]]\n        :type carpetLen: int\n        :rtype: int\n        '
        tiles.sort()
        prefix = [0] * (len(tiles) + 1)
        for (i, (l, r)) in enumerate(tiles):
            prefix[i + 1] = prefix[i] + (r - l + 1)
        result = 0
        for (left, (l, _)) in enumerate(tiles):
            r = l + carpetLen - 1
            right = bisect.bisect_right(tiles, [r + 1]) - 1
            extra = max(tiles[right][1] - r, 0)
            result = max(result, prefix[right + 1] - prefix[left] - extra)
        return result
import bisect

class Solution4(object):

    def maximumWhiteTiles(self, tiles, carpetLen):
        if False:
            print('Hello World!')
        '\n        :type tiles: List[List[int]]\n        :type carpetLen: int\n        :rtype: int\n        '
        tiles.sort()
        prefix = [0] * (len(tiles) + 1)
        for (i, (l, r)) in enumerate(tiles):
            prefix[i + 1] = prefix[i] + (r - l + 1)
        result = 0
        for (right, (_, r)) in enumerate(tiles):
            l = r - carpetLen + 1
            left = bisect.bisect_right(tiles, [l])
            if left - 1 >= 0 and tiles[left - 1][1] + 1 >= l:
                left -= 1
            extra = max(l - tiles[left][0], 0)
            result = max(result, prefix[right + 1] - prefix[left] - extra)
        return result