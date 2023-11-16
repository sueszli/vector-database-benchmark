from __future__ import annotations
from vowpalwabbit import pyvw
from river import base

class VW2RiverBase:

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.vw = pyvw.Workspace(*args, **kwargs)

    def _format_x(self, x):
        if False:
            i = 10
            return i + 15
        return ' '.join((f'{k}:{v}' for (k, v) in x.items()))

class VW2RiverClassifier(VW2RiverBase, base.Classifier):

    def learn_one(self, x, y):
        if False:
            return 10
        y = int(y)
        y_vw = 2 * y - 1
        ex = self._format_x(x)
        ex = f'{y_vw} | {ex}'
        self.vw.learn(ex)
        return self

    def predict_proba_one(self, x):
        if False:
            return 10
        ex = '| ' + self._format_x(x)
        y_pred = self.vw.predict(ex)
        return {True: y_pred, False: 1.0 - y_pred}