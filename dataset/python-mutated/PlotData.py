import numpy as np

class PlotData(object):
    """
    Class used for managing plot data
      - allows data sharing between multiple graphics items (curve, scatter, graph..)
      - each item may define the columns it needs
      - column groupings ('pos' or x, y, z)
      - efficiently appendable 
      - log, fft transformations
      - color mode conversion (float/byte/qcolor)
      - pen/brush conversion
      - per-field cached masking
        - allows multiple masking fields (different graphics need to mask on different criteria) 
        - removal of nan/inf values
      - option for single value shared by entire column
      - cached downsampling
      - cached min / max / hasnan / isuniform
    """

    def __init__(self):
        if False:
            return 10
        self.fields = {}
        self.maxVals = {}
        self.minVals = {}

    def addFields(self, **fields):
        if False:
            for i in range(10):
                print('nop')
        for f in fields:
            if f not in self.fields:
                self.fields[f] = None

    def hasField(self, f):
        if False:
            for i in range(10):
                print('nop')
        return f in self.fields

    def __getitem__(self, field):
        if False:
            i = 10
            return i + 15
        return self.fields[field]

    def __setitem__(self, field, val):
        if False:
            for i in range(10):
                print('nop')
        self.fields[field] = val

    def max(self, field):
        if False:
            return 10
        mx = self.maxVals.get(field, None)
        if mx is None:
            mx = np.max(self[field])
            self.maxVals[field] = mx
        return mx

    def min(self, field):
        if False:
            for i in range(10):
                print('nop')
        mn = self.minVals.get(field, None)
        if mn is None:
            mn = np.min(self[field])
            self.minVals[field] = mn
        return mn