import numpy as np
from skopt.space import Integer

class SKDecimal(Integer):

    def __init__(self, low, high, decimals=3, prior='uniform', base=10, transform=None, name=None, dtype=np.int64):
        if False:
            return 10
        self.decimals = decimals
        self.pow_dot_one = pow(0.1, self.decimals)
        self.pow_ten = pow(10, self.decimals)
        _low = int(low * self.pow_ten)
        _high = int(high * self.pow_ten)
        self.low_orig = round(_low * self.pow_dot_one, self.decimals)
        self.high_orig = round(_high * self.pow_dot_one, self.decimals)
        super().__init__(_low, _high, prior, base, transform, name, dtype)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return "Decimal(low={}, high={}, decimals={}, prior='{}', transform='{}')".format(self.low_orig, self.high_orig, self.decimals, self.prior, self.transform_)

    def __contains__(self, point):
        if False:
            while True:
                i = 10
        if isinstance(point, list):
            point = np.array(point)
        return self.low_orig <= point <= self.high_orig

    def transform(self, Xt):
        if False:
            print('Hello World!')
        return super().transform([int(v * self.pow_ten) for v in Xt])

    def inverse_transform(self, Xt):
        if False:
            for i in range(10):
                print('nop')
        res = super().inverse_transform(Xt)
        return [int(v) / self.pow_ten for v in res]