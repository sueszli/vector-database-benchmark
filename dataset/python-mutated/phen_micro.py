"""Classes to work with Phenotype Microarray data.

More information on the single plates can be found here: http://www.biolog.com/

Classes:
 - PlateRecord - Object that contain time course data on each well of the
   plate, as well as metadata (if any).
 - WellRecord - Object that contains the time course data of a single well
 - JsonWriter - Writer of PlateRecord objects in JSON format.

Functions:
 - JsonIterator -  Incremental PM JSON parser, this is an iterator that returns
   PlateRecord objects.
 - CsvIterator - Incremental PM CSV parser, this is an iterator that returns
   PlateRecord objects.
 - _toOPM - Used internally by JsonWriter, converts PlateRecord objects in
   dictionaries ready to be serialized in JSON format.

"""
import warnings
import json
import csv
import numpy as np
from Bio import BiopythonParserWarning
_datafile = 'Data File'
_plate = 'Plate Type'
_strainType = 'Strain Type'
_sample = 'Sample Number'
_strainName = 'Strain Name'
_strainNumber = 'Strain Number'
_other = 'Other'
_hour = 'Hour'
_file = 'File'
_position = 'Position'
_setupTime = 'Setup Time'
_platesPrefix = 'PM'
_platesPrefixMammalian = 'PM-M'
_csvData = 'csv_data'
_measurements = 'measurements'

class PlateRecord:
    """PlateRecord object for storing Phenotype Microarray plates data.

    A PlateRecord stores all the wells of a particular phenotype
    Microarray plate, along with metadata (if any). The single wells can be
    accessed calling their id as an index or iterating on the PlateRecord:

    >>> from Bio import phenotype
    >>> plate = phenotype.read("phenotype/Plate.json", "pm-json")
    >>> well = plate['A05']
    >>> for well in plate:
    ...    print(well.id)
    ...
    A01
    ...

    The plate rows and columns can be queried with an indexing system similar
    to NumPy and other matrices:

    >>> print(plate[1])
    Plate ID: PM01
    Well: 12
    Rows: 1
    Columns: 12
    PlateRecord('WellRecord['B01'], WellRecord['B02'], WellRecord['B03'], ..., WellRecord['B12']')

    >>> print(plate[:,1])
    Plate ID: PM01
    Well: 8
    Rows: 8
    Columns: 1
    PlateRecord('WellRecord['A02'], WellRecord['B02'], WellRecord['C02'], ..., WellRecord['H02']')

    Single WellRecord objects can be accessed using this indexing system:

    >>> print(plate[1,2])
    Plate ID: PM01
    Well ID: B03
    Time points: 384
    Minum signal 0.00 at time 11.00
    Maximum signal 76.25 at time 18.00
    WellRecord('(0.0, 11.0), (0.25, 11.0), (0.5, 11.0), (0.75, 11.0), (1.0, 11.0), ..., (95.75, 11.0)')

    The presence of a particular well can be inspected with the "in" keyword:
    >>> 'A01' in plate
    True

    All the wells belonging to a "row" (identified by the first character of
    the well id) in the plate can be obtained:

    >>> for well in plate.get_row('H'):
    ...     print(well.id)
    ...
    H01
    H02
    H03
    ...

    All the wells belonging to a "column" (identified by the number of the well)
    in the plate can be obtained:

    >>> for well in plate.get_column(12):
    ...     print(well.id)
    ...
    A12
    B12
    C12
    ...

    Two PlateRecord objects can be compared: if all their wells are equal the
    two plates are considered equal:

    >>> plate2 = phenotype.read("phenotype/Plate.json", "pm-json")
    >>> plate == plate2
    True

    Two PlateRecord object can be summed up or subtracted from each other: the
    the signals of each well will be summed up or subtracted. The id of the
    left operand will be kept:

    >>> plate3 = plate + plate2
    >>> print(plate3.id)
    PM01

    Many Phenotype Microarray plate have a "negative control" well, which can
    be subtracted to all wells:

    >>> subplate = plate.subtract_control()

    """

    def __init__(self, plateid, wells=None):
        if False:
            return 10
        'Initialize the class.'
        self.id = plateid
        if wells is None:
            wells = []
        self.qualifiers = {}
        self._wells = {}
        try:
            for w in wells:
                self._is_well(w)
                self[w.id] = w
        except TypeError:
            raise TypeError('You must provide an iterator-like object containing the single wells')
        self._update()

    def _update(self):
        if False:
            while True:
                i = 10
        'Update the rows and columns string identifiers (PRIVATE).'
        self._rows = sorted({x[0] for x in self._wells})
        self._columns = sorted({x[1:] for x in self._wells})

    def _is_well(self, obj):
        if False:
            print('Hello World!')
        'Check if the given object is a WellRecord object (PRIVATE).\n\n        Used both for the class constructor and the __setitem__ method\n        '
        if not isinstance(obj, WellRecord):
            raise ValueError(f'A WellRecord type object is needed as value (got {type(obj)})')

    def __getitem__(self, index):
        if False:
            print('Hello World!')
        'Access part of the plate.\n\n        Depending on the indices, you can get a WellRecord object\n        (representing a single well of the plate),\n        or another plate\n        (representing some part or all of the original plate).\n\n        plate[wid] gives a WellRecord (if wid is a WellRecord id)\n        plate[r,c] gives a WellRecord\n        plate[r] gives a row as a PlateRecord\n        plate[r,:] gives a row as a PlateRecord\n        plate[:,c] gives a column as a PlateRecord\n\n        plate[:] and plate[:,:] give a copy of the plate\n\n        Anything else gives a subset of the original plate, e.g.\n        plate[0:2] or plate[0:2,:] uses only row 0 and 1\n        plate[:,1:3] uses only columns 1 and 2\n        plate[0:2,1:3] uses only rows 0 & 1 and only cols 1 & 2\n\n        >>> from Bio import phenotype\n        >>> plate = phenotype.read("phenotype/Plate.json", "pm-json")\n\n        You can access a well of the plate, using its id.\n\n        >>> w = plate[\'A01\']\n\n        You can access a row of the plate as a PlateRecord using an integer\n        index:\n\n        >>> first_row = plate[0]\n        >>> print(first_row)\n        Plate ID: PM01\n        Well: 12\n        Rows: 1\n        Columns: 12\n        PlateRecord(\'WellRecord[\'A01\'], WellRecord[\'A02\'], WellRecord[\'A03\'], ..., WellRecord[\'A12\']\')\n        >>> last_row = plate[-1]\n        >>> print(last_row)\n        Plate ID: PM01\n        Well: 12\n        Rows: 1\n        Columns: 12\n        PlateRecord(\'WellRecord[\'H01\'], WellRecord[\'H02\'], WellRecord[\'H03\'], ..., WellRecord[\'H12\']\')\n\n        You can also access use python\'s slice notation to sub-plates\n        containing only some of the plate rows:\n\n        >>> sub_plate = plate[2:5]\n        >>> print(sub_plate)\n        Plate ID: PM01\n        Well: 36\n        Rows: 3\n        Columns: 12\n        PlateRecord(\'WellRecord[\'C01\'], WellRecord[\'C02\'], WellRecord[\'C03\'], ..., WellRecord[\'E12\']\')\n\n        This includes support for a step, i.e. plate[start:end:step], which\n        can be used to select every second row:\n\n        >>> sub_plate = plate[::2]\n\n        You can also use two indices to specify both rows and columns.\n        Using simple integers gives you the single wells. e.g.\n\n        >>> w = plate[3, 4]\n        >>> print(w.id)\n        D05\n\n        To get a single column use this syntax:\n\n        >>> sub_plate = plate[:, 4]\n        >>> print(sub_plate)\n        Plate ID: PM01\n        Well: 8\n        Rows: 8\n        Columns: 1\n        PlateRecord(\'WellRecord[\'A05\'], WellRecord[\'B05\'], WellRecord[\'C05\'], ..., WellRecord[\'H05\']\')\n\n        Or, to get part of a column,\n\n        >>> sub_plate = plate[1:3, 4]\n        >>> print(sub_plate)\n        Plate ID: PM01\n        Well: 2\n        Rows: 2\n        Columns: 1\n        PlateRecord(WellRecord[\'B05\'], WellRecord[\'C05\'])\n\n        However, in general you get a sub-plate,\n\n        >>> print(plate[1:5, 3:6])\n        Plate ID: PM01\n        Well: 12\n        Rows: 4\n        Columns: 3\n        PlateRecord(\'WellRecord[\'B04\'], WellRecord[\'B05\'], WellRecord[\'B06\'], ..., WellRecord[\'E06\']\')\n\n        This should all seem familiar to anyone who has used the NumPy\n        array or matrix objects.\n        '
        if isinstance(index, str):
            try:
                return self._wells[index]
            except KeyError:
                raise KeyError(f'Well {index} not found!')
        elif isinstance(index, int):
            try:
                row = self._rows[index]
            except IndexError:
                raise IndexError('Row %d not found!' % index)
            return PlateRecord(self.id, filter(lambda x: x.id.startswith(row), self._wells.values()))
        elif isinstance(index, slice):
            rows = self._rows[index]
            return PlateRecord(self.id, filter(lambda x: x.id[0] in rows, self._wells.values()))
        elif len(index) != 2:
            raise TypeError('Invalid index type.')
        (row_index, col_index) = index
        if isinstance(row_index, int) and isinstance(col_index, int):
            try:
                row = self._rows[row_index]
            except IndexError:
                raise IndexError('Row %d not found!' % row_index)
            try:
                col = self._columns[col_index]
            except IndexError:
                raise IndexError('Column %d not found!' % col_index)
            return self._wells[row + col]
        elif isinstance(row_index, int):
            try:
                row = self._rows[row_index]
            except IndexError:
                raise IndexError('Row %d not found!' % row_index)
            cols = self._columns[col_index]
            return PlateRecord(self.id, filter(lambda x: x.id.startswith(row) and x.id[1:] in cols, self._wells.values()))
        elif isinstance(col_index, int):
            try:
                col = self._columns[col_index]
            except IndexError:
                raise IndexError('Columns %d not found!' % col_index)
            rows = self._rows[row_index]
            return PlateRecord(self.id, filter(lambda x: x.id.endswith(col) and x.id[0] in rows, self._wells.values()))
        else:
            rows = self._rows[row_index]
            cols = self._columns[col_index]
            return PlateRecord(self.id, filter(lambda x: x.id[0] in rows and x.id[1:] in cols, self._wells.values()))

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        if not isinstance(key, str):
            raise ValueError('Well identifier should be string-like')
        self._is_well(value)
        if value.id != key:
            raise ValueError("WellRecord ID and provided key are different (got '%s' and '%s')" % (type(value.id), type(key)))
        self._wells[key] = value
        self._update()

    def __delitem__(self, key):
        if False:
            return 10
        if not isinstance(key, str):
            raise ValueError('Well identifier should be string-like')
        del self._wells[key]
        self._update()

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        for well in sorted(self._wells):
            yield self._wells[well]

    def __contains__(self, wellid):
        if False:
            print('Hello World!')
        if wellid in self._wells:
            return True
        return False

    def __len__(self):
        if False:
            print('Hello World!')
        'Return the number of wells in this plate.'
        return len(self._wells)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, self.__class__):
            return self._wells == other._wells
        else:
            return False

    def __add__(self, plate):
        if False:
            print('Hello World!')
        'Add another PlateRecord object.\n\n        The wells in both plates must be the same\n\n        A new PlateRecord object is returned, having the same id as the\n        left operand.\n        '
        if not isinstance(plate, PlateRecord):
            raise TypeError('Expecting a PlateRecord object')
        if {x.id for x in self} != {x.id for x in plate}:
            raise ValueError('The two plates have different wells')
        wells = []
        for w in self:
            wells.append(w + plate[w.id])
        newp = PlateRecord(self.id, wells=wells)
        return newp

    def __sub__(self, plate):
        if False:
            i = 10
            return i + 15
        'Subtract another PlateRecord object.\n\n        The wells in both plates must be the same\n\n        A new PlateRecord object is returned, having the same id as the\n        left operand.\n        '
        if not isinstance(plate, PlateRecord):
            raise TypeError('Expecting a PlateRecord object')
        if {x.id for x in self} != {x.id for x in plate}:
            raise ValueError('The two plates have different wells')
        wells = []
        for w in self:
            wells.append(w - plate[w.id])
        newp = PlateRecord(self.id, wells=wells)
        return newp

    def get_row(self, row):
        if False:
            while True:
                i = 10
        "Get all the wells of a given row.\n\n        A row is identified with a letter (e.g. 'A')\n        "
        try:
            row = str(row)
        except Exception:
            raise ValueError('Row identifier should be string-like')
        if len(row) > 1:
            raise ValueError('Row identifier must be of maximum one letter')
        for w in sorted(filter(lambda x: x.startswith(row), self._wells)):
            yield self._wells[w]

    def get_column(self, column):
        if False:
            print('Hello World!')
        "Get all the wells of a given column.\n\n        A column is identified with a number (e.g. '6')\n        "
        try:
            column = int(column)
        except Exception:
            raise ValueError('Column identifier should be a number')
        for w in sorted(filter(lambda x: x.endswith('%02d' % column), self._wells)):
            yield self._wells[w]

    def subtract_control(self, control='A01', wells=None):
        if False:
            for i in range(10):
                print('nop')
        "Subtract a 'control' well from the other plates wells.\n\n        By default the control is subtracted to all wells, unless\n        a list of well ID is provided\n\n        The control well should belong to the plate\n        A new PlateRecord object is returned\n        "
        if control not in self:
            raise ValueError('Control well not present in plate')
        wcontrol = self[control]
        if wells is None:
            wells = self._wells.keys()
        missing = {w for w in wells if w not in self}
        if missing:
            raise ValueError('Some wells to be subtracted are not present')
        nwells = []
        for w in self:
            if w.id in wells:
                nwells.append(w - wcontrol)
            else:
                nwells.append(w)
        newp = PlateRecord(self.id, wells=nwells)
        return newp

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'Return a (truncated) representation of the plate for debugging.'
        if len(self._wells) > 4:
            return "%s('%s, ..., %s')" % (self.__class__.__name__, ', '.join(["%s['%s']" % (self[x].__class__.__name__, self[x].id) for x in sorted(self._wells.keys())[:3]]), "%s['%s']" % (self[sorted(self._wells.keys())[-1]].__class__.__name__, self[sorted(self._wells.keys())[-1]].id))
        else:
            return '%s(%s)' % (self.__class__.__name__, ', '.join(["%s['%s']" % (self[x].__class__.__name__, self[x].id) for x in sorted(self._wells.keys())]))

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a human readable summary of the record (string).\n\n        The python built in function str works by calling the object\'s __str__\n        method.  e.g.\n\n        >>> from Bio import phenotype\n        >>> record = next(phenotype.parse("phenotype/Plates.csv", "pm-csv"))\n        >>> print(record)\n        Plate ID: PM01\n        Well: 96\n        Rows: 8\n        Columns: 12\n        PlateRecord(\'WellRecord[\'A01\'], WellRecord[\'A02\'], WellRecord[\'A03\'], ..., WellRecord[\'H12\']\')\n\n        Note that long well lists are shown truncated.\n        '
        lines = []
        if self.id:
            lines.append(f'Plate ID: {self.id}')
        lines.append('Well: %i' % len(self))
        lines.append('Rows: %d' % len({x.id[0] for x in self}))
        lines.append('Columns: %d' % len({x.id[1:3] for x in self}))
        lines.append(repr(self))
        return '\n'.join(lines)

class WellRecord:
    """WellRecord stores all time course signals of a phenotype Microarray well.

    The single time points and signals can be accessed iterating on the
    WellRecord or using lists indexes or slices:

    >>> from Bio import phenotype
    >>> plate = phenotype.read("phenotype/Plate.json", "pm-json")
    >>> well = plate['A05']
    >>> for time, signal in well:
    ...    print("Time: %f, Signal: %f" % (time, signal)) # doctest:+ELLIPSIS
    ...
    Time: 0.000000, Signal: 14.000000
    Time: 0.250000, Signal: 13.000000
    Time: 0.500000, Signal: 15.000000
    Time: 0.750000, Signal: 15.000000
    ...
    >>> well[1]
    16.0
    >>> well[1:5]
    [16.0, 20.0, 18.0, 15.0]
    >>> well[1:5:0.5]
    [16.0, 19.0, 20.0, 18.0, 18.0, 18.0, 15.0, 18.0]

    If a time point was not present in the input file but it's between the
    minimum and maximum time point, the interpolated signal is returned,
    otherwise a nan value:

    >>> well[1.3]
    19.0
    >>> well[1250]
    nan

    Two WellRecord objects can be compared: if their input time/signal pairs
    are exactly the same, the two records are considered equal:

    >>> well2 = plate['H12']
    >>> well == well2
    False

    Two WellRecord objects can be summed up or subtracted from each other: a new
    WellRecord object is returned, having the left operand id.

    >>> well1 = plate['A05']
    >>> well2 = well + well1
    >>> print(well2.id)
    A05

    If SciPy is installed, a sigmoid function can be fitted to the PM curve,
    in order to extract some parameters; three sigmoid functions are available:
    * gompertz
    * logistic
    * richards
    The functions are described in Zwietering et al., 1990 (PMID: 16348228)

    For example::

        well.fit()
        print(well.slope, well.model)
        (61.853516785566917, 'logistic')

    If not sigmoid function is specified, the first one that is successfully
    fitted is used. The user can also specify a specific function.

    To specify gompertz::

        well.fit('gompertz')
        print(well.slope, well.model)
        (127.94630059171354, 'gompertz')

    If no function can be fitted, the parameters are left as None, except for
    the max, min, average_height and area.
    """

    def __init__(self, wellid, plate=None, signals=None):
        if False:
            print('Hello World!')
        'Initialize the class.'
        if plate is None:
            self.plate = PlateRecord(None)
        else:
            self.plate = plate
        self.id = wellid
        self.max = None
        self.min = None
        self.average_height = None
        self.area = None
        self.plateau = None
        self.slope = None
        self.lag = None
        self.v = None
        self.y0 = None
        self.model = None
        if signals is None:
            self._signals = {}
        else:
            self._signals = signals

    def _interpolate(self, time):
        if False:
            return 10
        'Linear interpolation of the signals at certain time points (PRIVATE).'
        times = sorted(self._signals.keys())
        return np.interp(time, times, [self._signals[x] for x in times], left=np.nan, right=np.nan)

    def __setitem__(self, time, signal):
        if False:
            print('Hello World!')
        'Assign a signal at a certain time point.'
        try:
            time = float(time)
        except ValueError:
            raise ValueError('Time point should be a number')
        try:
            signal = float(signal)
        except ValueError:
            raise ValueError('Signal should be a number')
        self._signals[time] = signal

    def __getitem__(self, time):
        if False:
            while True:
                i = 10
        'Return a subset of signals or a single signal.'
        if isinstance(time, slice):
            if time.start is None:
                start = 0
            else:
                start = time.start
            if time.stop is None:
                stop = max(self.get_times())
            else:
                stop = time.stop
            time = np.arange(start, stop, time.step)
            return list(self._interpolate(time))
        elif isinstance(time, (float, int)):
            return self._interpolate(time)
        raise ValueError('Invalid index')

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        for time in sorted(self._signals.keys()):
            yield (time, self._signals[time])

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, self.__class__):
            if list(self._signals.keys()) != list(other._signals.keys()):
                return False
            for k in self._signals:
                if np.isnan(self[k]) and np.isnan(other[k]):
                    continue
                elif self[k] != other[k]:
                    return False
            return True
        else:
            return False

    def __add__(self, well):
        if False:
            print('Hello World!')
        'Add another WellRecord object.\n\n        A new WellRecord object is returned, having the same id as the\n        left operand\n        '
        if not isinstance(well, WellRecord):
            raise TypeError('Expecting a WellRecord object')
        signals = {}
        times = set(self._signals.keys()).union(set(well._signals.keys()))
        for t in sorted(times):
            signals[t] = self[t] + well[t]
        neww = WellRecord(self.id, signals=signals)
        return neww

    def __sub__(self, well):
        if False:
            for i in range(10):
                print('nop')
        'Subtract another WellRecord object.\n\n        A new WellRecord object is returned, having the same id as the\n        left operand\n        '
        if not isinstance(well, WellRecord):
            raise TypeError('Expecting a WellRecord object')
        signals = {}
        times = set(self._signals.keys()).union(set(well._signals.keys()))
        for t in sorted(times):
            signals[t] = self[t] - well[t]
        neww = WellRecord(self.id, signals=signals)
        return neww

    def __len__(self):
        if False:
            return 10
        'Return the number of time points sampled.'
        return len(self._signals)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a (truncated) representation of the signals for debugging.'
        if len(self) > 7:
            return "%s('%s, ..., %s')" % (self.__class__.__name__, ', '.join([str(x) for x in self.get_raw()[:5]]), str(self.get_raw()[-1]))
        else:
            return '%s(%s)' % (self.__class__.__name__, ', '.join([str(x) for x in self.get_raw()]))

    def __str__(self):
        if False:
            print('Hello World!')
        'Return a human readable summary of the record (string).\n\n        The python built-in function str works by calling the object\'s __str__\n        method.  e.g.\n\n        >>> from Bio import phenotype\n        >>> plate = phenotype.read("phenotype/Plate.json", "pm-json")\n        >>> record = plate[\'A05\']\n        >>> print(record)\n        Plate ID: PM01\n        Well ID: A05\n        Time points: 384\n        Minum signal 0.25 at time 13.00\n        Maximum signal 19.50 at time 23.00\n        WellRecord(\'(0.0, 14.0), (0.25, 13.0), (0.5, 15.0), (0.75, 15.0), (1.0, 16.0), ..., (95.75, 16.0)\')\n\n        Note that long time spans are shown truncated.\n        '
        lines = []
        if self.plate and self.plate.id:
            lines.append(f'Plate ID: {self.plate.id}')
        if self.id:
            lines.append(f'Well ID: {self.id}')
        lines.append('Time points: %i' % len(self))
        lines.append('Minum signal %.2f at time %.2f' % min(self, key=lambda x: x[1]))
        lines.append('Maximum signal %.2f at time %.2f' % max(self, key=lambda x: x[1]))
        lines.append(repr(self))
        return '\n'.join(lines)

    def get_raw(self):
        if False:
            while True:
                i = 10
        'Get a list of time/signal pairs.'
        return [(t, self._signals[t]) for t in sorted(self._signals.keys())]

    def get_times(self):
        if False:
            i = 10
            return i + 15
        'Get a list of the recorded time points.'
        return sorted(self._signals.keys())

    def get_signals(self):
        if False:
            print('Hello World!')
        'Get a list of the recorded signals (ordered by collection time).'
        return [self._signals[t] for t in sorted(self._signals.keys())]

    def fit(self, function=('gompertz', 'logistic', 'richards')):
        if False:
            return 10
        "Fit a sigmoid function to this well and extract curve parameters.\n\n        If function is None or an empty tuple/list, then no fitting is done.\n        Only the object's ``.min``, ``.max`` and ``.average_height`` are\n        calculated.\n\n        By default the following fitting functions will be used in order:\n         - gompertz\n         - logistic\n         - richards\n\n        The first function that is successfully fitted to the signals will\n        be used to extract the curve parameters and update ``.area`` and\n        ``.model``. If no function can be fitted an exception is raised.\n\n        The function argument should be a tuple or list of any of these three\n        function names as strings.\n\n        There is no return value.\n        "
        avail_func = ('gompertz', 'logistic', 'richards')
        self.max = max(self, key=lambda x: x[1])[1]
        self.min = min(self, key=lambda x: x[1])[1]
        self.average_height = np.array(self.get_signals()).mean()
        if not function:
            self.area = None
            self.model = None
            return
        for sigmoid_func in function:
            if sigmoid_func not in avail_func:
                raise ValueError(f'Fitting function {sigmoid_func!r} not supported')
        from .pm_fitting import fit, get_area
        from .pm_fitting import logistic, gompertz, richards
        function_map = {'logistic': logistic, 'gompertz': gompertz, 'richards': richards}
        self.area = get_area(self.get_signals(), self.get_times())
        self.model = None
        for sigmoid_func in function:
            func = function_map[sigmoid_func]
            try:
                ((self.plateau, self.slope, self.lag, self.v, self.y0), pcov) = fit(func, self.get_times(), self.get_signals())
                self.model = sigmoid_func
                return
            except RuntimeError:
                continue
        raise RuntimeError('Could not fit any sigmoid function')

def JsonIterator(handle):
    if False:
        print('Hello World!')
    'Iterate over PM json records as PlateRecord objects.\n\n    Arguments:\n     - handle - input file\n\n    '
    try:
        data = json.load(handle)
    except ValueError:
        raise ValueError('Could not parse JSON file')
    if hasattr(data, 'keys'):
        data = [data]
    for pobj in data:
        try:
            plateID = pobj[_csvData][_plate]
        except TypeError:
            raise TypeError('Malformed JSON input')
        except KeyError:
            raise KeyError('Could not retrieve plate id')
        if not plateID.startswith(_platesPrefix) and (not plateID.startswith(_platesPrefixMammalian)):
            warnings.warn(f'Non-standard plate ID found ({plateID})', BiopythonParserWarning)
        else:
            if plateID.startswith(_platesPrefixMammalian):
                pID = plateID[len(_platesPrefixMammalian):]
            else:
                pID = plateID[len(_platesPrefix):]
            while len(pID) > 0:
                try:
                    int(pID)
                    break
                except ValueError:
                    pID = pID[:-1]
            if len(pID) == 0:
                warnings.warn(f'Non-standard plate ID found ({plateID})', BiopythonParserWarning)
            elif int(pID) < 0:
                warnings.warn(f'Non-standard plate ID found ({plateID}), using {_platesPrefix}{abs(int(pID))}')
                plateID = _platesPrefix + str(abs(int(pID)))
            elif plateID.startswith(_platesPrefixMammalian):
                plateID = _platesPrefixMammalian + '%02d' % int(pID)
            else:
                plateID = _platesPrefix + '%02d' % int(pID)
        try:
            times = pobj[_measurements][_hour]
        except KeyError:
            raise KeyError('Could not retrieve the time points')
        plate = PlateRecord(plateID)
        for k in pobj[_measurements]:
            if k == _hour:
                continue
            plate[k] = WellRecord(k, plate=plate, signals={times[i]: pobj[_measurements][k][i] for i in range(len(times))})
        del pobj['measurements']
        plate.qualifiers = pobj
        yield plate

def CsvIterator(handle):
    if False:
        i = 10
        return i + 15
    'Iterate over PM csv records as PlateRecord objects.\n\n    Arguments:\n     - handle - input file\n\n    '
    plate = None
    data = False
    qualifiers = {}
    idx = {}
    wells = {}
    tblreader = csv.reader(handle, delimiter=',', quotechar='"')
    for line in tblreader:
        if len(line) < 2:
            continue
        elif _datafile in line[0].strip():
            if plate is not None:
                qualifiers[_csvData][_datafile] = line[1].strip()
                plate = PlateRecord(plate.id)
                for (k, v) in wells.items():
                    plate[k] = WellRecord(k, plate, v)
                plate.qualifiers = qualifiers
                yield plate
            plate = PlateRecord(None)
            data = False
            qualifiers[_csvData] = {}
            idx = {}
            wells = {}
        elif _plate in line[0].strip():
            plateID = line[1].strip()
            qualifiers[_csvData][_plate] = plateID
            if not plateID.startswith(_platesPrefix) and (not plateID.startswith(_platesPrefixMammalian)):
                warnings.warn(f'Non-standard plate ID found ({plateID})', BiopythonParserWarning)
            else:
                if plateID.startswith(_platesPrefixMammalian):
                    pID = plateID[len(_platesPrefixMammalian):]
                else:
                    pID = plateID[len(_platesPrefix):]
                while len(pID) > 0:
                    try:
                        int(pID)
                        break
                    except ValueError:
                        pID = pID[:-1]
                if len(pID) == 0:
                    warnings.warn(f'Non-standard plate ID found ({plateID})', BiopythonParserWarning)
                elif int(pID) < 0:
                    warnings.warn(f'Non-standard plate ID found ({plateID}), using {_platesPrefix}{abs(int(pID))}')
                    plateID = _platesPrefix + str(abs(int(pID)))
                elif plateID.startswith(_platesPrefixMammalian):
                    plateID = _platesPrefixMammalian + '%02d' % int(pID)
                else:
                    plateID = _platesPrefix + '%02d' % int(pID)
            plate.id = plateID
        elif _strainType in line[0].strip():
            if plate is None:
                continue
            qualifiers[_csvData][_strainType] = line[1].strip()
        elif _sample in line[0].strip():
            if plate is None:
                continue
            qualifiers[_csvData][_sample] = line[1].strip()
        elif _strainNumber in line[0].strip():
            if plate is None:
                continue
            qualifiers[_csvData][_strainNumber] = line[1].strip()
        elif _strainName in line[0].strip():
            if plate is None:
                continue
            qualifiers[_csvData][_strainName] = line[1].strip()
        elif _other in line[0].strip():
            if plate is None:
                continue
            qualifiers[_csvData][_other] = line[1].strip()
        elif _file in line[0].strip():
            if plate is None:
                continue
            qualifiers[_csvData][_file] = line[1].strip()
        elif _position in line[0].strip():
            if plate is None:
                continue
            qualifiers[_csvData][_position] = line[1].strip()
        elif _setupTime in line[0].strip():
            if plate is None:
                continue
            qualifiers[_csvData][_setupTime] = line[1].strip()
        elif _hour in line[0].strip():
            if plate is None:
                continue
            data = True
            for i in range(1, len(line)):
                x = line[i]
                if x == '':
                    continue
                wells[x.strip()] = {}
                idx[i] = x.strip()
        elif data:
            if plate is None:
                continue
            try:
                float(line[0])
            except ValueError:
                continue
            time = float(line[0])
            for i in range(1, len(line)):
                x = line[i]
                try:
                    signal = float(x)
                except ValueError:
                    continue
                well = idx[i]
                wells[well][time] = signal
    if plate is not None and plate.id is not None:
        plate = PlateRecord(plate.id)
        for (k, v) in wells.items():
            plate[k] = WellRecord(k, plate, v)
        plate.qualifiers = qualifiers
        yield plate

def _toOPM(plate):
    if False:
        for i in range(10):
            print('nop')
    'Transform a PlateRecord object into a dictionary (PRIVATE).'
    d = dict(plate.qualifiers.items())
    d[_csvData] = {}
    d[_csvData][_plate] = plate.id
    d[_measurements] = {}
    d[_measurements][_hour] = []
    times = set()
    for (wid, w) in plate._wells.items():
        d[_measurements][wid] = []
        for hour in w._signals:
            times.add(hour)
    for hour in sorted(times):
        d[_measurements][_hour].append(hour)
        for (wid, w) in plate._wells.items():
            if hour in w._signals:
                d[_measurements][wid].append(w[hour])
            else:
                d[_measurements][wid].append(float('nan'))
    return d

class JsonWriter:
    """Class to write PM Json format files."""

    def __init__(self, plates):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        self.plates = plates

    def write(self, handle):
        if False:
            while True:
                i = 10
        "Write this instance's plates to a file handle."
        out = []
        for plate in self.plates:
            try:
                out.append(_toOPM(plate))
            except ValueError:
                raise ValueError('Could not export plate(s) in JSON format')
        handle.write(json.dumps(out) + '\n')
        return len(out)
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest(verbose=0)