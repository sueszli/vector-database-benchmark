import water.udf.CMetricFunc as MetricFunc

class MAE(MetricFunc):

    def map(self, pred, act, w, o, model):
        if False:
            return 10
        return [abs(pred[0] - act[0]), 1]

    def reduce(self, l, r):
        if False:
            for i in range(10):
                print('nop')
        return [l[0] + r[0], l[1] + r[1]]

    def metric(self, l):
        if False:
            for i in range(10):
                print('nop')
        return l[0] / l[1]