"""
==========
Evans test
==========

A mockup "Foo" units class which supports conversion and different tick
formatting depending on the "unit".  Here the "unit" is just a scalar
conversion factor, but this example shows that Matplotlib is entirely agnostic
to what kind of units client packages use.
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.units as units

class Foo:

    def __init__(self, val, unit=1.0):
        if False:
            i = 10
            return i + 15
        self.unit = unit
        self._val = val * unit

    def value(self, unit):
        if False:
            i = 10
            return i + 15
        if unit is None:
            unit = self.unit
        return self._val / unit

class FooConverter(units.ConversionInterface):

    @staticmethod
    def axisinfo(unit, axis):
        if False:
            while True:
                i = 10
        'Return the Foo AxisInfo.'
        if unit == 1.0 or unit == 2.0:
            return units.AxisInfo(majloc=ticker.IndexLocator(8, 0), majfmt=ticker.FormatStrFormatter('VAL: %s'), label='foo')
        else:
            return None

    @staticmethod
    def convert(obj, unit, axis):
        if False:
            i = 10
            return i + 15
        '\n        Convert *obj* using *unit*.\n\n        If *obj* is a sequence, return the converted sequence.\n        '
        if np.iterable(obj):
            return [o.value(unit) for o in obj]
        else:
            return obj.value(unit)

    @staticmethod
    def default_units(x, axis):
        if False:
            for i in range(10):
                print('nop')
        'Return the default unit for *x* or None.'
        if np.iterable(x):
            for thisx in x:
                return thisx.unit
        else:
            return x.unit
units.registry[Foo] = FooConverter()
x = [Foo(val, 1.0) for val in range(0, 50, 2)]
y = [i for i in range(len(x))]
(fig, (ax1, ax2)) = plt.subplots(1, 2)
fig.suptitle('Custom units')
fig.subplots_adjust(bottom=0.2)
ax2.plot(x, y, 'o', xunits=2.0)
ax2.set_title('xunits = 2.0')
plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
ax1.plot(x, y)
ax1.set_title('default units')
plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
plt.show()