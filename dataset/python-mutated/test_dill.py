import dill
from ignite.metrics import Metric

class Accumulation(Metric):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.value = 0
        super(Accumulation, self).__init__()

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.value = 0

    def compute(self):
        if False:
            print('Hello World!')
        return self.value

    def update(self, output):
        if False:
            for i in range(10):
                print('nop')
        self.value += output

def test_metric():
    if False:
        return 10

    def _test(m, values, e):
        if False:
            print('Hello World!')
        for v in values:
            m.update(v)
        assert m.compute() == e
    metric = Accumulation()
    m1 = dill.loads(dill.dumps(metric))
    values = list(range(10))
    expected = sum(values)
    _test(m1, values, expected)
    metric.update(5)
    m2 = dill.loads(dill.dumps(metric))
    _test(m2, values, expected + 5)