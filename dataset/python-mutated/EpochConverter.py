"""EpochConverter module containing class EpochConverter."""
from matplotlib import cbook, units
import matplotlib.dates as date_ticker
__all__ = ['EpochConverter']

class EpochConverter(units.ConversionInterface):
    """
    Provides Matplotlib conversion functionality for Monte Epoch and Duration
    classes.
    """
    jdRef = 1721425.5 - 1

    @staticmethod
    def axisinfo(unit, axis):
        if False:
            return 10
        majloc = date_ticker.AutoDateLocator()
        majfmt = date_ticker.AutoDateFormatter(majloc)
        return units.AxisInfo(majloc=majloc, majfmt=majfmt, label=unit)

    @staticmethod
    def float2epoch(value, unit):
        if False:
            while True:
                i = 10
        '\n        Convert a Matplotlib floating-point date into an Epoch of the specified\n        units.\n\n        = INPUT VARIABLES\n        - value     The Matplotlib floating-point date.\n        - unit      The unit system to use for the Epoch.\n\n        = RETURN VALUE\n        - Returns the value converted to an Epoch in the specified time system.\n        '
        import matplotlib.testing.jpl_units as U
        secPastRef = value * 86400.0 * U.UnitDbl(1.0, 'sec')
        return U.Epoch(unit, secPastRef, EpochConverter.jdRef)

    @staticmethod
    def epoch2float(value, unit):
        if False:
            return 10
        '\n        Convert an Epoch value to a float suitable for plotting as a python\n        datetime object.\n\n        = INPUT VARIABLES\n        - value    An Epoch or list of Epochs that need to be converted.\n        - unit     The units to use for an axis with Epoch data.\n\n        = RETURN VALUE\n        - Returns the value parameter converted to floats.\n        '
        return value.julianDate(unit) - EpochConverter.jdRef

    @staticmethod
    def duration2float(value):
        if False:
            print('Hello World!')
        '\n        Convert a Duration value to a float suitable for plotting as a python\n        datetime object.\n\n        = INPUT VARIABLES\n        - value    A Duration or list of Durations that need to be converted.\n\n        = RETURN VALUE\n        - Returns the value parameter converted to floats.\n        '
        return value.seconds() / 86400.0

    @staticmethod
    def convert(value, unit, axis):
        if False:
            print('Hello World!')
        import matplotlib.testing.jpl_units as U
        if not cbook.is_scalar_or_string(value):
            return [EpochConverter.convert(x, unit, axis) for x in value]
        if unit is None:
            unit = EpochConverter.default_units(value, axis)
        if isinstance(value, U.Duration):
            return EpochConverter.duration2float(value)
        else:
            return EpochConverter.epoch2float(value, unit)

    @staticmethod
    def default_units(value, axis):
        if False:
            while True:
                i = 10
        if cbook.is_scalar_or_string(value):
            return value.frame()
        else:
            return EpochConverter.default_units(value[0], axis)