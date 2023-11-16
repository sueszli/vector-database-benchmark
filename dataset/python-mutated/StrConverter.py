"""StrConverter module containing class StrConverter."""
import numpy as np
import matplotlib.units as units
__all__ = ['StrConverter']

class StrConverter(units.ConversionInterface):
    """
    A Matplotlib converter class for string data values.

    Valid units for string are:
    - 'indexed' : Values are indexed as they are specified for plotting.
    - 'sorted'  : Values are sorted alphanumerically.
    - 'inverted' : Values are inverted so that the first value is on top.
    - 'sorted-inverted' :  A combination of 'sorted' and 'inverted'
    """

    @staticmethod
    def axisinfo(unit, axis):
        if False:
            i = 10
            return i + 15
        return None

    @staticmethod
    def convert(value, unit, axis):
        if False:
            for i in range(10):
                print('nop')
        if value == []:
            return []
        ax = axis.axes
        if axis is ax.xaxis:
            isXAxis = True
        else:
            isXAxis = False
        axis.get_major_ticks()
        ticks = axis.get_ticklocs()
        labels = axis.get_ticklabels()
        labels = [l.get_text() for l in labels if l.get_text()]
        if not labels:
            ticks = []
            labels = []
        if not np.iterable(value):
            value = [value]
        newValues = []
        for v in value:
            if v not in labels and v not in newValues:
                newValues.append(v)
        labels.extend(newValues)
        labels = [''] + labels + ['']
        ticks = list(range(len(labels)))
        ticks[0] = 0.5
        ticks[-1] = ticks[-1] - 0.5
        axis.set_ticks(ticks)
        axis.set_ticklabels(labels)
        loc = axis.get_major_locator()
        loc.set_bounds(ticks[0], ticks[-1])
        if isXAxis:
            ax.set_xlim(ticks[0], ticks[-1])
        else:
            ax.set_ylim(ticks[0], ticks[-1])
        result = [ticks[labels.index(v)] for v in value]
        ax.viewLim.ignore(-1)
        return result

    @staticmethod
    def default_units(value, axis):
        if False:
            for i in range(10):
                print('nop')
        return 'indexed'