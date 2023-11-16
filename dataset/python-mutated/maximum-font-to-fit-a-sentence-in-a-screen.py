import collections

class FontInfo(object):

    def getWidth(self, fontSize, ch):
        if False:
            return 10
        '\n        :type fontSize: int\n        :type ch: char\n        :rtype int\n        '
        pass

    def getHeight(self, fontSize):
        if False:
            while True:
                i = 10
        '\n        :type fontSize: int\n        :rtype int\n        '
        pass

class Solution(object):

    def maxFont(self, text, w, h, fonts, fontInfo):
        if False:
            print('Hello World!')
        '\n        :type text: str\n        :type w: int\n        :type h: int\n        :type fonts: List[int]\n        :type fontInfo: FontInfo\n        :rtype: int\n        '

        def check(count, w, h, fonts, fontInfo, x):
            if False:
                return 10
            return fontInfo.getHeight(fonts[x]) <= h and sum((cnt * fontInfo.getWidth(fonts[x], c) for (c, cnt) in count.iteritems())) <= w
        count = collections.Counter(text)
        (left, right) = (0, len(fonts) - 1)
        while left <= right:
            mid = left + (right - left) // 2
            if not check(count, w, h, fonts, fontInfo, mid):
                right = mid - 1
            else:
                left = mid + 1
        return fonts[right] if right >= 0 else -1