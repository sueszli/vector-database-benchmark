import functools
import math
import operator

class Stat:

    def __init__(self, image_or_list, mask=None):
        if False:
            return 10
        try:
            if mask:
                self.h = image_or_list.histogram(mask)
            else:
                self.h = image_or_list.histogram()
        except AttributeError:
            self.h = image_or_list
        if not isinstance(self.h, list):
            msg = 'first argument must be image or list'
            raise TypeError(msg)
        self.bands = list(range(len(self.h) // 256))

    def __getattr__(self, id):
        if False:
            return 10
        'Calculate missing attribute'
        if id[:4] == '_get':
            raise AttributeError(id)
        v = getattr(self, '_get' + id)()
        setattr(self, id, v)
        return v

    def _getextrema(self):
        if False:
            i = 10
            return i + 15
        'Get min/max values for each band in the image'

        def minmax(histogram):
            if False:
                for i in range(10):
                    print('nop')
            n = 255
            x = 0
            for i in range(256):
                if histogram[i]:
                    n = min(n, i)
                    x = max(x, i)
            return (n, x)
        v = []
        for i in range(0, len(self.h), 256):
            v.append(minmax(self.h[i:]))
        return v

    def _getcount(self):
        if False:
            for i in range(10):
                print('nop')
        'Get total number of pixels in each layer'
        v = []
        for i in range(0, len(self.h), 256):
            v.append(functools.reduce(operator.add, self.h[i:i + 256]))
        return v

    def _getsum(self):
        if False:
            return 10
        'Get sum of all pixels in each layer'
        v = []
        for i in range(0, len(self.h), 256):
            layer_sum = 0.0
            for j in range(256):
                layer_sum += j * self.h[i + j]
            v.append(layer_sum)
        return v

    def _getsum2(self):
        if False:
            return 10
        'Get squared sum of all pixels in each layer'
        v = []
        for i in range(0, len(self.h), 256):
            sum2 = 0.0
            for j in range(256):
                sum2 += j ** 2 * float(self.h[i + j])
            v.append(sum2)
        return v

    def _getmean(self):
        if False:
            return 10
        'Get average pixel level for each layer'
        v = []
        for i in self.bands:
            v.append(self.sum[i] / self.count[i])
        return v

    def _getmedian(self):
        if False:
            for i in range(10):
                print('nop')
        'Get median pixel level for each layer'
        v = []
        for i in self.bands:
            s = 0
            half = self.count[i] // 2
            b = i * 256
            for j in range(256):
                s = s + self.h[b + j]
                if s > half:
                    break
            v.append(j)
        return v

    def _getrms(self):
        if False:
            i = 10
            return i + 15
        'Get RMS for each layer'
        v = []
        for i in self.bands:
            v.append(math.sqrt(self.sum2[i] / self.count[i]))
        return v

    def _getvar(self):
        if False:
            return 10
        'Get variance for each layer'
        v = []
        for i in self.bands:
            n = self.count[i]
            v.append((self.sum2[i] - self.sum[i] ** 2.0 / n) / n)
        return v

    def _getstddev(self):
        if False:
            i = 10
            return i + 15
        'Get standard deviation for each layer'
        v = []
        for i in self.bands:
            v.append(math.sqrt(self.var[i]))
        return v
Global = Stat