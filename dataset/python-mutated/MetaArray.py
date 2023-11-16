"""
MetaArray.py -  Class encapsulating ndarray with meta data
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.

MetaArray is an array class based on numpy.ndarray that allows storage of per-axis meta data
such as axis values, names, units, column names, etc. It also enables several
new methods for slicing and indexing the array based on this meta data. 
More info at http://www.scipy.org/Cookbook/MetaArray
"""
import copy
import os
import pickle
import warnings
import numpy as np
USE_HDF5 = True
try:
    import h5py
    if not hasattr(h5py, 'Group'):
        import h5py.highlevel
        h5py.Group = h5py.highlevel.Group
        h5py.Dataset = h5py.highlevel.Dataset
    HAVE_HDF5 = True
except:
    USE_HDF5 = False
    HAVE_HDF5 = False

def axis(name=None, cols=None, values=None, units=None):
    if False:
        return 10
    'Convenience function for generating axis descriptions when defining MetaArrays'
    ax = {}
    cNameOrder = ['name', 'units', 'title']
    if name is not None:
        ax['name'] = name
    if values is not None:
        ax['values'] = values
    if units is not None:
        ax['units'] = units
    if cols is not None:
        ax['cols'] = []
        for c in cols:
            if type(c) != list and type(c) != tuple:
                c = [c]
            col = {}
            for i in range(0, len(c)):
                col[cNameOrder[i]] = c[i]
            ax['cols'].append(col)
    return ax

class sliceGenerator(object):
    """Just a compact way to generate tuples of slice objects."""

    def __getitem__(self, arg):
        if False:
            return 10
        return arg

    def __getslice__(self, arg):
        if False:
            print('Hello World!')
        return arg
SLICER = sliceGenerator()

class MetaArray(object):
    """N-dimensional array with meta data such as axis titles, units, and column names.
  
    May be initialized with a file name, a tuple representing the dimensions of the array,
    or any arguments that could be passed on to numpy.array()
  
    The info argument sets the metadata for the entire array. It is composed of a list
    of axis descriptions where each axis may have a name, title, units, and a list of column 
    descriptions. An additional dict at the end of the axis list may specify parameters
    that apply to values in the entire array.
  
    For example:
        A 2D array of altitude values for a topographical map might look like
            info=[
        {'name': 'lat', 'title': 'Lattitude'}, 
        {'name': 'lon', 'title': 'Longitude'}, 
        {'title': 'Altitude', 'units': 'm'}
      ]
        In this case, every value in the array represents the altitude in feet at the lat, lon
        position represented by the array index. All of the following return the 
        value at lat=10, lon=5:
            array[10, 5]
            array['lon':5, 'lat':10]
            array['lat':10][5]
        Now suppose we want to combine this data with another array of equal dimensions that
        represents the average rainfall for each location. We could easily store these as two 
        separate arrays or combine them into a 3D array with this description:
            info=[
        {'name': 'vals', 'cols': [
          {'name': 'altitude', 'units': 'm'}, 
          {'name': 'rainfall', 'units': 'cm/year'}
        ]},
        {'name': 'lat', 'title': 'Lattitude'}, 
        {'name': 'lon', 'title': 'Longitude'}
      ]
        We can now access the altitude values with array[0] or array['altitude'], and the
        rainfall values with array[1] or array['rainfall']. All of the following return
        the rainfall value at lat=10, lon=5:
            array[1, 10, 5]
            array['lon':5, 'lat':10, 'val': 'rainfall']
            array['rainfall', 'lon':5, 'lat':10]
        Notice that in the second example, there is no need for an extra (4th) axis description
        since the actual values are described (name and units) in the column info for the first axis.
    """
    version = u'2'
    defaultCompression = None
    nameTypes = [str, tuple]

    @staticmethod
    def isNameType(var):
        if False:
            for i in range(10):
                print('nop')
        return any((isinstance(var, t) for t in MetaArray.nameTypes))
    wrapMethods = set(['__eq__', '__ne__', '__le__', '__lt__', '__ge__', '__gt__'])

    def __init__(self, data=None, info=None, dtype=None, file=None, copy=False, **kwargs):
        if False:
            i = 10
            return i + 15
        object.__init__(self)
        warnings.warn('MetaArray is deprecated and will be removed in 0.14. Available though https://pypi.org/project/MetaArray/ as its own package.', DeprecationWarning, stacklevel=2)
        self._isHDF = False
        if file is not None:
            self._data = None
            self.readFile(file, **kwargs)
            if kwargs.get('readAllData', True) and self._data is None:
                raise Exception('File read failed: %s' % file)
        else:
            self._info = info
            if hasattr(data, 'implements') and data.implements('MetaArray'):
                self._info = data._info
                self._data = data.asarray()
            elif isinstance(data, tuple):
                self._data = np.empty(data, dtype=dtype)
            else:
                self._data = np.array(data, dtype=dtype, copy=copy)
        self.checkInfo()

    def checkInfo(self):
        if False:
            for i in range(10):
                print('nop')
        info = self._info
        if info is None:
            if self._data is None:
                return
            else:
                self._info = [{} for i in range(self.ndim + 1)]
                return
        else:
            try:
                info = list(info)
            except:
                raise Exception('Info must be a list of axis specifications')
            if len(info) < self.ndim + 1:
                info.extend([{}] * (self.ndim + 1 - len(info)))
            elif len(info) > self.ndim + 1:
                raise Exception('Info parameter must be list of length ndim+1 or less.')
            for i in range(len(info)):
                if not isinstance(info[i], dict):
                    if info[i] is None:
                        info[i] = {}
                    else:
                        raise Exception('Axis specification must be Dict or None')
                if i < self.ndim and 'values' in info[i]:
                    if type(info[i]['values']) is list:
                        info[i]['values'] = np.array(info[i]['values'])
                    elif type(info[i]['values']) is not np.ndarray:
                        raise Exception('Axis values must be specified as list or ndarray')
                    if info[i]['values'].ndim != 1 or info[i]['values'].shape[0] != self.shape[i]:
                        raise Exception('Values array for axis %d has incorrect shape. (given %s, but should be %s)' % (i, str(info[i]['values'].shape), str((self.shape[i],))))
                if i < self.ndim and 'cols' in info[i]:
                    if not isinstance(info[i]['cols'], list):
                        info[i]['cols'] = list(info[i]['cols'])
                    if len(info[i]['cols']) != self.shape[i]:
                        raise Exception('Length of column list for axis %d does not match data. (given %d, but should be %d)' % (i, len(info[i]['cols']), self.shape[i]))
            self._info = info

    def implements(self, name=None):
        if False:
            return 10
        if name is None:
            return ['MetaArray']
        else:
            return name == 'MetaArray'

    def __getitem__(self, ind):
        if False:
            i = 10
            return i + 15
        nInd = self._interpretIndexes(ind)
        a = self._data[nInd]
        if len(nInd) == self.ndim:
            if np.all([not isinstance(ind, (slice, np.ndarray)) for ind in nInd]):
                return a
        info = []
        extraInfo = self._info[-1].copy()
        for i in range(0, len(nInd)):
            if type(nInd[i]) in [slice, list] or isinstance(nInd[i], np.ndarray):
                info.append(self._axisSlice(i, nInd[i]))
            else:
                newInfo = self._axisSlice(i, nInd[i])
                name = None
                colName = None
                for k in newInfo:
                    if k == 'cols':
                        if 'cols' not in extraInfo:
                            extraInfo['cols'] = []
                        extraInfo['cols'].append(newInfo[k])
                        if 'units' in newInfo[k]:
                            extraInfo['units'] = newInfo[k]['units']
                        if 'name' in newInfo[k]:
                            colName = newInfo[k]['name']
                    elif k == 'name':
                        name = newInfo[k]
                    else:
                        if k not in extraInfo:
                            extraInfo[k] = newInfo[k]
                        extraInfo[k] = newInfo[k]
                if 'name' not in extraInfo:
                    if name is None:
                        if colName is not None:
                            extraInfo['name'] = colName
                    elif colName is not None:
                        extraInfo['name'] = str(name) + ': ' + str(colName)
                    else:
                        extraInfo['name'] = name
        info.append(extraInfo)
        return MetaArray(a, info=info)

    @property
    def ndim(self):
        if False:
            return 10
        return len(self.shape)

    @property
    def shape(self):
        if False:
            print('Hello World!')
        return self._data.shape

    @property
    def dtype(self):
        if False:
            return 10
        return self._data.dtype

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._data)

    def __getslice__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self.__getitem__(slice(*args))

    def __setitem__(self, ind, val):
        if False:
            i = 10
            return i + 15
        nInd = self._interpretIndexes(ind)
        try:
            self._data[nInd] = val
        except:
            print(self, nInd, val)
            raise

    def __getattr__(self, attr):
        if False:
            for i in range(10):
                print('nop')
        if attr in self.wrapMethods:
            return getattr(self._data, attr)
        else:
            raise AttributeError(attr)

    def __eq__(self, b):
        if False:
            i = 10
            return i + 15
        return self._binop('__eq__', b)

    def __ne__(self, b):
        if False:
            i = 10
            return i + 15
        return self._binop('__ne__', b)

    def __sub__(self, b):
        if False:
            while True:
                i = 10
        return self._binop('__sub__', b)

    def __add__(self, b):
        if False:
            while True:
                i = 10
        return self._binop('__add__', b)

    def __mul__(self, b):
        if False:
            print('Hello World!')
        return self._binop('__mul__', b)

    def __div__(self, b):
        if False:
            for i in range(10):
                print('nop')
        return self._binop('__div__', b)

    def __truediv__(self, b):
        if False:
            i = 10
            return i + 15
        return self._binop('__truediv__', b)

    def _binop(self, op, b):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(b, MetaArray):
            b = b.asarray()
        a = self.asarray()
        c = getattr(a, op)(b)
        if c.shape != a.shape:
            raise Exception('Binary operators with MetaArray must return an array of the same shape (this shape is %s, result shape was %s)' % (a.shape, c.shape))
        return MetaArray(c, info=self.infoCopy())

    def asarray(self):
        if False:
            while True:
                i = 10
        if isinstance(self._data, np.ndarray):
            return self._data
        else:
            return np.array(self._data)

    def __array__(self, dtype=None):
        if False:
            i = 10
            return i + 15
        if dtype is None:
            return self.asarray()
        else:
            return self.asarray().astype(dtype)

    def axisValues(self, axis):
        if False:
            print('Hello World!')
        'Return the list of values for an axis'
        ax = self._interpretAxis(axis)
        if 'values' in self._info[ax]:
            return self._info[ax]['values']
        else:
            raise Exception('Array axis %s (%d) has no associated values.' % (str(axis), ax))

    def xvals(self, axis):
        if False:
            while True:
                i = 10
        'Synonym for axisValues()'
        return self.axisValues(axis)

    def axisHasValues(self, axis):
        if False:
            for i in range(10):
                print('nop')
        ax = self._interpretAxis(axis)
        return 'values' in self._info[ax]

    def axisHasColumns(self, axis):
        if False:
            for i in range(10):
                print('nop')
        ax = self._interpretAxis(axis)
        return 'cols' in self._info[ax]

    def axisUnits(self, axis):
        if False:
            print('Hello World!')
        'Return the units for axis'
        ax = self._info[self._interpretAxis(axis)]
        if 'units' in ax:
            return ax['units']

    def hasColumn(self, axis, col):
        if False:
            while True:
                i = 10
        ax = self._info[self._interpretAxis(axis)]
        if 'cols' in ax:
            for c in ax['cols']:
                if c['name'] == col:
                    return True
        return False

    def listColumns(self, axis=None):
        if False:
            print('Hello World!')
        'Return a list of column names for axis. If axis is not specified, then return a dict of {axisName: (column names), ...}.'
        if axis is None:
            ret = {}
            for i in range(self.ndim):
                if 'cols' in self._info[i]:
                    cols = [c['name'] for c in self._info[i]['cols']]
                else:
                    cols = []
                ret[self.axisName(i)] = cols
            return ret
        else:
            axis = self._interpretAxis(axis)
            return [c['name'] for c in self._info[axis]['cols']]

    def columnName(self, axis, col):
        if False:
            i = 10
            return i + 15
        ax = self._info[self._interpretAxis(axis)]
        return ax['cols'][col]['name']

    def axisName(self, n):
        if False:
            print('Hello World!')
        return self._info[n].get('name', n)

    def columnUnits(self, axis, column):
        if False:
            for i in range(10):
                print('nop')
        'Return the units for column in axis'
        ax = self._info[self._interpretAxis(axis)]
        if 'cols' in ax:
            for c in ax['cols']:
                if c['name'] == column:
                    return c['units']
            raise Exception('Axis %s has no column named %s' % (str(axis), str(column)))
        else:
            raise Exception('Axis %s has no column definitions' % str(axis))

    def rowsort(self, axis, key=0):
        if False:
            while True:
                i = 10
        'Return this object with all records sorted along axis using key as the index to the values to compare. Does not yet modify meta info.'
        keyList = self[key]
        order = keyList.argsort()
        if type(axis) == int:
            ind = [slice(None)] * axis
            ind.append(order)
        elif isinstance(axis, str):
            ind = (slice(axis, order),)
        else:
            raise TypeError('axis must be type (int, str)')
        return self[tuple(ind)]

    def append(self, val, axis):
        if False:
            return 10
        'Return this object with val appended along axis. Does not yet combine meta info.'
        s = list(self.shape)
        axis = self._interpretAxis(axis)
        s[axis] += 1
        n = MetaArray(tuple(s), info=self._info, dtype=self.dtype)
        ind = [slice(None)] * self.ndim
        ind[axis] = slice(None, -1)
        n[tuple(ind)] = self
        ind[axis] = -1
        n[tuple(ind)] = val
        return n

    def extend(self, val, axis):
        if False:
            i = 10
            return i + 15
        'Return the concatenation along axis of this object and val. Does not yet combine meta info.'
        axis = self._interpretAxis(axis)
        return MetaArray(np.concatenate(self, val, axis), info=self._info)

    def infoCopy(self, axis=None):
        if False:
            i = 10
            return i + 15
        'Return a deep copy of the axis meta info for this object'
        if axis is None:
            return copy.deepcopy(self._info)
        else:
            return copy.deepcopy(self._info[self._interpretAxis(axis)])

    def copy(self):
        if False:
            print('Hello World!')
        return MetaArray(self._data.copy(), info=self.infoCopy())

    def _interpretIndexes(self, ind):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(ind, tuple):
            if isinstance(ind, list) and len(ind) > 0 and isinstance(ind[0], slice):
                ind = tuple(ind)
            else:
                ind = (ind,)
        nInd = [slice(None)] * self.ndim
        numOk = True
        for i in range(0, len(ind)):
            (axis, index, isNamed) = self._interpretIndex(ind[i], i, numOk)
            nInd[axis] = index
            if isNamed:
                numOk = False
        return tuple(nInd)

    def _interpretAxis(self, axis):
        if False:
            i = 10
            return i + 15
        if isinstance(axis, (str, tuple)):
            return self._getAxis(axis)
        else:
            return axis

    def _interpretIndex(self, ind, pos, numOk):
        if False:
            print('Hello World!')
        if type(ind) is int:
            if not numOk:
                raise Exception('string and integer indexes may not follow named indexes')
            return (pos, ind, False)
        if MetaArray.isNameType(ind):
            if not numOk:
                raise Exception('string and integer indexes may not follow named indexes')
            return (pos, self._getIndex(pos, ind), False)
        elif type(ind) is slice:
            if MetaArray.isNameType(ind.start) or MetaArray.isNameType(ind.stop):
                axis = self._interpretAxis(ind.start)
                if MetaArray.isNameType(ind.stop):
                    index = self._getIndex(axis, ind.stop)
                elif (isinstance(ind.stop, float) or isinstance(ind.step, float)) and 'values' in self._info[axis]:
                    if ind.stop is None:
                        mask = self.xvals(axis) < ind.step
                    elif ind.step is None:
                        mask = self.xvals(axis) >= ind.stop
                    else:
                        mask = (self.xvals(axis) >= ind.stop) * (self.xvals(axis) < ind.step)
                    index = mask
                elif isinstance(ind.stop, int) or isinstance(ind.step, int):
                    if ind.step is None:
                        index = ind.stop
                    else:
                        index = slice(ind.stop, ind.step)
                elif type(ind.stop) is list:
                    index = []
                    for i in ind.stop:
                        if type(i) is int:
                            index.append(i)
                        elif MetaArray.isNameType(i):
                            index.append(self._getIndex(axis, i))
                        else:
                            index = ind.stop
                            break
                else:
                    index = ind.stop
                return (axis, index, True)
            else:
                return (pos, ind, False)
        elif type(ind) is list:
            indList = [self._interpretIndex(i, pos, numOk)[1] for i in ind]
            return (pos, indList, False)
        else:
            if not numOk:
                raise Exception('string and integer indexes may not follow named indexes')
            return (pos, ind, False)

    def _getAxis(self, name):
        if False:
            print('Hello World!')
        for i in range(0, len(self._info)):
            axis = self._info[i]
            if 'name' in axis and axis['name'] == name:
                return i
        raise Exception('No axis named %s.\n  info=%s' % (name, self._info))

    def _getIndex(self, axis, name):
        if False:
            while True:
                i = 10
        ax = self._info[axis]
        if ax is not None and 'cols' in ax:
            for i in range(0, len(ax['cols'])):
                if 'name' in ax['cols'][i] and ax['cols'][i]['name'] == name:
                    return i
        raise Exception('Axis %d has no column named %s.\n  info=%s' % (axis, name, self._info))

    def _axisCopy(self, i):
        if False:
            print('Hello World!')
        return copy.deepcopy(self._info[i])

    def _axisSlice(self, i, cols):
        if False:
            i = 10
            return i + 15
        if 'cols' in self._info[i] or 'values' in self._info[i]:
            ax = self._axisCopy(i)
            if 'cols' in ax:
                sl = np.array(ax['cols'])[cols]
                if isinstance(sl, np.ndarray):
                    sl = list(sl)
                ax['cols'] = sl
            if 'values' in ax:
                ax['values'] = np.array(ax['values'])[cols]
        else:
            ax = self._info[i]
        return ax

    def prettyInfo(self):
        if False:
            while True:
                i = 10
        s = ''
        titles = []
        maxl = 0
        for i in range(len(self._info) - 1):
            ax = self._info[i]
            axs = ''
            if 'name' in ax:
                axs += '"%s"' % str(ax['name'])
            else:
                axs += '%d' % i
            if 'units' in ax:
                axs += ' (%s)' % str(ax['units'])
            titles.append(axs)
            if len(axs) > maxl:
                maxl = len(axs)
        for i in range(min(self.ndim, len(self._info) - 1)):
            ax = self._info[i]
            axs = titles[i]
            axs += '%s[%d] :' % (' ' * (maxl - len(axs) + 5 - len(str(self.shape[i]))), self.shape[i])
            if 'values' in ax:
                if self.shape[i] > 0:
                    v0 = ax['values'][0]
                    axs += '  values: [%g' % v0
                    if self.shape[i] > 1:
                        v1 = ax['values'][-1]
                        axs += ' ... %g] (step %g)' % (v1, (v1 - v0) / (self.shape[i] - 1))
                    else:
                        axs += ']'
                else:
                    axs += '  values: []'
            if 'cols' in ax:
                axs += ' columns: '
                colstrs = []
                for c in range(len(ax['cols'])):
                    col = ax['cols'][c]
                    cs = str(col.get('name', c))
                    if 'units' in col:
                        cs += ' (%s)' % col['units']
                    colstrs.append(cs)
                axs += '[' + ', '.join(colstrs) + ']'
            s += axs + '\n'
        s += str(self._info[-1])
        return s

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '%s\n-----------------------------------------------\n%s' % (self.view(np.ndarray).__repr__(), self.prettyInfo())

    def __str__(self):
        if False:
            print('Hello World!')
        return self.__repr__()

    def axisCollapsingFn(self, fn, axis=None, *args, **kargs):
        if False:
            print('Hello World!')
        fn = getattr(self._data, fn)
        if axis is None:
            return fn(axis, *args, **kargs)
        else:
            info = self.infoCopy()
            axis = self._interpretAxis(axis)
            info.pop(axis)
            return MetaArray(fn(axis, *args, **kargs), info=info)

    def mean(self, axis=None, *args, **kargs):
        if False:
            i = 10
            return i + 15
        return self.axisCollapsingFn('mean', axis, *args, **kargs)

    def min(self, axis=None, *args, **kargs):
        if False:
            print('Hello World!')
        return self.axisCollapsingFn('min', axis, *args, **kargs)

    def max(self, axis=None, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        return self.axisCollapsingFn('max', axis, *args, **kargs)

    def transpose(self, *args):
        if False:
            return 10
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            order = args[0]
        else:
            order = args
        order = [self._interpretAxis(ax) for ax in order]
        infoOrder = order + list(range(len(order), len(self._info)))
        info = [self._info[i] for i in infoOrder]
        order = order + list(range(len(order), self.ndim))
        try:
            if self._isHDF:
                return MetaArray(np.array(self._data).transpose(order), info=info)
            else:
                return MetaArray(self._data.transpose(order), info=info)
        except:
            print(order)
            raise

    def readFile(self, filename, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Load the data and meta info stored in *filename*\n        Different arguments are allowed depending on the type of file.\n        For HDF5 files:\n        \n            *writable* (bool) if True, then any modifications to data in the array will be stored to disk.\n            *readAllData* (bool) if True, then all data in the array is immediately read from disk\n                          and the file is closed (this is the default for files < 500MB). Otherwise, the file will\n                          be left open and data will be read only as requested (this is \n                          the default for files >= 500MB).\n        \n        \n        '
        with open(filename, 'rb') as fd:
            magic = fd.read(8)
            if magic == b'\x89HDF\r\n\x1a\n':
                fd.close()
                self._readHDF5(filename, **kwargs)
                self._isHDF = True
            else:
                fd.seek(0)
                meta = MetaArray._readMeta(fd)
                if not kwargs.get('readAllData', True):
                    self._data = np.empty(meta['shape'], dtype=meta['type'])
                if 'version' in meta:
                    ver = meta['version']
                else:
                    ver = 1
                rFuncName = '_readData%s' % str(ver)
                if not hasattr(MetaArray, rFuncName):
                    raise Exception("This MetaArray library does not support array version '%s'" % ver)
                rFunc = getattr(self, rFuncName)
                rFunc(fd, meta, **kwargs)
                self._isHDF = False

    @staticmethod
    def _readMeta(fd):
        if False:
            print('Hello World!')
        'Read meta array from the top of a file. Read lines until a blank line is reached.\n        This function should ideally work for ALL versions of MetaArray.\n        '
        meta = u''
        while True:
            line = fd.readline().strip()
            if line == '':
                break
            meta += line
        ret = eval(meta)
        return ret

    def _readData1(self, fd, meta, mmap=False, **kwds):
        if False:
            return 10
        frameSize = 1
        for ax in meta['info']:
            if 'values_len' in ax:
                ax['values'] = np.frombuffer(fd.read(ax['values_len']), dtype=ax['values_type'])
                frameSize *= ax['values_len']
                del ax['values_len']
                del ax['values_type']
        self._info = meta['info']
        if not kwds.get('readAllData', True):
            return
        if mmap:
            subarr = np.memmap(fd, dtype=meta['type'], mode='r', shape=meta['shape'])
        else:
            subarr = np.frombuffer(fd.read(), dtype=meta['type'])
            subarr.shape = meta['shape']
        self._data = subarr

    def _readData2(self, fd, meta, mmap=False, subset=None, **kwds):
        if False:
            while True:
                i = 10
        dynAxis = None
        frameSize = 1
        for i in range(len(meta['info'])):
            ax = meta['info'][i]
            if 'values_len' in ax:
                if ax['values_len'] == 'dynamic':
                    if dynAxis is not None:
                        raise Exception('MetaArray has more than one dynamic axis! (this is not allowed)')
                    dynAxis = i
                else:
                    ax['values'] = np.frombuffer(fd.read(ax['values_len']), dtype=ax['values_type'])
                    frameSize *= ax['values_len']
                    del ax['values_len']
                    del ax['values_type']
        self._info = meta['info']
        if not kwds.get('readAllData', True):
            return
        if dynAxis is None:
            if meta['type'] == 'object':
                if mmap:
                    raise Exception('memmap not supported for arrays with dtype=object')
                subarr = pickle.loads(fd.read())
            elif mmap:
                subarr = np.memmap(fd, dtype=meta['type'], mode='r', shape=meta['shape'])
            else:
                subarr = np.frombuffer(fd.read(), dtype=meta['type'])
            subarr.shape = meta['shape']
        else:
            if mmap:
                raise Exception('memmap not supported for non-contiguous arrays. Use rewriteContiguous() to convert.')
            ax = meta['info'][dynAxis]
            xVals = []
            frames = []
            frameShape = list(meta['shape'])
            frameShape[dynAxis] = 1
            frameSize = np.prod(frameShape)
            n = 0
            while True:
                while True:
                    line = fd.readline()
                    if line != '\n':
                        break
                if line == '':
                    break
                inf = eval(line)
                if meta['type'] == 'object':
                    data = pickle.loads(fd.read(inf['len']))
                else:
                    data = np.frombuffer(fd.read(inf['len']), dtype=meta['type'])
                if data.size != frameSize * inf['numFrames']:
                    raise Exception('Wrong frame size in MetaArray file! (frame %d)' % n)
                shape = list(frameShape)
                shape[dynAxis] = inf['numFrames']
                data.shape = shape
                if subset is not None:
                    dSlice = subset[dynAxis]
                    if dSlice.start is None:
                        dStart = 0
                    else:
                        dStart = max(0, dSlice.start - n)
                    if dSlice.stop is None:
                        dStop = data.shape[dynAxis]
                    else:
                        dStop = min(data.shape[dynAxis], dSlice.stop - n)
                    newSubset = list(subset[:])
                    newSubset[dynAxis] = slice(dStart, dStop)
                    if dStop > dStart:
                        frames.append(data[tuple(newSubset)].copy())
                else:
                    frames.append(data)
                n += inf['numFrames']
                if 'xVals' in inf:
                    xVals.extend(inf['xVals'])
            subarr = np.concatenate(frames, axis=dynAxis)
            if len(xVals) > 0:
                ax['values'] = np.array(xVals, dtype=ax['values_type'])
            del ax['values_len']
            del ax['values_type']
        self._info = meta['info']
        self._data = subarr

    def _readHDF5(self, fileName, readAllData=None, writable=False, **kargs):
        if False:
            while True:
                i = 10
        if 'close' in kargs and readAllData is None:
            readAllData = kargs['close']
        if readAllData is True and writable is True:
            raise Exception('Incompatible arguments: readAllData=True and writable=True')
        if not HAVE_HDF5:
            try:
                assert writable == False
                assert readAllData != False
                self._readHDF5Remote(fileName)
                return
            except:
                raise Exception("The file '%s' is HDF5-formatted, but the HDF5 library (h5py) was not found." % fileName)
        if readAllData is None:
            size = os.stat(fileName).st_size
            readAllData = size < 500000000.0
        if writable is True:
            mode = 'r+'
        else:
            mode = 'r'
        f = h5py.File(fileName, mode)
        ver = f.attrs['MetaArray']
        try:
            ver = ver.decode('utf-8')
        except:
            pass
        if ver > MetaArray.version:
            print('Warning: This file was written with MetaArray version %s, but you are using version %s. (Will attempt to read anyway)' % (str(ver), str(MetaArray.version)))
        meta = MetaArray.readHDF5Meta(f['info'])
        self._info = meta
        if writable or not readAllData:
            self._data = f['data']
            self._openFile = f
        else:
            self._data = f['data'][:]
            f.close()

    def _readHDF5Remote(self, fileName):
        if False:
            print('Hello World!')
        proc = getattr(MetaArray, '_hdf5Process', None)
        if proc == False:
            raise Exception('remote read failed')
        if proc is None:
            from .. import multiprocess as mp
            proc = mp.Process(executable='/usr/bin/python')
            proc.setProxyOptions(deferGetattr=True)
            MetaArray._hdf5Process = proc
            MetaArray._h5py_metaarray = proc._import('pyqtgraph.metaarray')
        ma = MetaArray._h5py_metaarray.MetaArray(file=fileName)
        self._data = ma.asarray()._getValue()
        self._info = ma._info._getValue()

    @staticmethod
    def mapHDF5Array(data, writable=False):
        if False:
            return 10
        off = data.id.get_offset()
        if writable:
            mode = 'r+'
        else:
            mode = 'r'
        if off is None:
            raise Exception('This dataset uses chunked storage; it can not be memory-mapped. (store using mappable=True)')
        return np.memmap(filename=data.file.filename, offset=off, dtype=data.dtype, shape=data.shape, mode=mode)

    @staticmethod
    def readHDF5Meta(root, mmap=False):
        if False:
            while True:
                i = 10
        data = {}
        for k in root.attrs:
            val = root.attrs[k]
            if isinstance(val, bytes):
                val = val.decode()
            if isinstance(val, str):
                try:
                    val = eval(val)
                except:
                    raise Exception('Can not evaluate string: "%s"' % val)
            data[k] = val
        for k in root:
            obj = root[k]
            if isinstance(obj, h5py.Group):
                val = MetaArray.readHDF5Meta(obj)
            elif isinstance(obj, h5py.Dataset):
                if mmap:
                    val = MetaArray.mapHDF5Array(obj)
                else:
                    val = obj[:]
            else:
                raise Exception("Don't know what to do with type '%s'" % str(type(obj)))
            data[k] = val
        typ = root.attrs['_metaType_']
        try:
            typ = typ.decode('utf-8')
        except:
            pass
        del data['_metaType_']
        if typ == 'dict':
            return data
        elif typ == 'list' or typ == 'tuple':
            d2 = [None] * len(data)
            for k in data:
                d2[int(k)] = data[k]
            if typ == 'tuple':
                d2 = tuple(d2)
            return d2
        else:
            raise Exception("Don't understand metaType '%s'" % typ)

    def write(self, fileName, **opts):
        if False:
            return 10
        'Write this object to a file. The object can be restored by calling MetaArray(file=fileName)\n        opts:\n            appendAxis: the name (or index) of the appendable axis. Allows the array to grow.\n            appendKeys: a list of keys (other than "values") for metadata to append to on the appendable axis.\n            compression: None, \'gzip\' (good compression), \'lzf\' (fast compression), etc.\n            chunks: bool or tuple specifying chunk shape\n        '
        if USE_HDF5 is False:
            return self.writeMa(fileName, **opts)
        elif HAVE_HDF5 is True:
            return self.writeHDF5(fileName, **opts)
        else:
            raise Exception('h5py is required for writing .ma hdf5 files, but it could not be imported.')

    def writeMeta(self, fileName):
        if False:
            while True:
                i = 10
        'Used to re-write meta info to the given file.\n        This feature is only available for HDF5 files.'
        f = h5py.File(fileName, 'r+')
        if f.attrs['MetaArray'] != MetaArray.version:
            raise Exception('The file %s was created with a different version of MetaArray. Will not modify.' % fileName)
        del f['info']
        self.writeHDF5Meta(f, 'info', self._info)
        f.close()

    def writeHDF5(self, fileName, **opts):
        if False:
            for i in range(10):
                print('nop')
        comp = self.defaultCompression
        if isinstance(comp, tuple):
            (comp, copts) = comp
        else:
            copts = None
        dsOpts = {'compression': comp, 'chunks': True}
        if copts is not None:
            dsOpts['compression_opts'] = copts
        appAxis = opts.get('appendAxis', None)
        if appAxis is not None:
            appAxis = self._interpretAxis(appAxis)
            cs = [min(100000, x) for x in self.shape]
            cs[appAxis] = 1
            dsOpts['chunks'] = tuple(cs)
        else:
            cs = [min(100000, x) for x in self.shape]
            for i in range(self.ndim):
                if 'cols' in self._info[i]:
                    cs[i] = 1
            dsOpts['chunks'] = tuple(cs)
        for k in dsOpts:
            if k in opts:
                dsOpts[k] = opts[k]
        if opts.get('mappable', False):
            dsOpts = {'chunks': None, 'compression': None}
        append = False
        if appAxis is not None:
            maxShape = list(self.shape)
            ax = self._interpretAxis(appAxis)
            maxShape[ax] = None
            if os.path.exists(fileName):
                append = True
            dsOpts['maxshape'] = tuple(maxShape)
        else:
            dsOpts['maxshape'] = None
        if append:
            f = h5py.File(fileName, 'r+')
            if f.attrs['MetaArray'] != MetaArray.version:
                raise Exception('The file %s was created with a different version of MetaArray. Will not modify.' % fileName)
            data = f['data']
            shape = list(data.shape)
            shape[ax] += self.shape[ax]
            data.resize(tuple(shape))
            sl = [slice(None)] * len(data.shape)
            sl[ax] = slice(-self.shape[ax], None)
            data[tuple(sl)] = self.view(np.ndarray)
            axKeys = ['values']
            axKeys.extend(opts.get('appendKeys', []))
            axInfo = f['info'][str(ax)]
            for key in axKeys:
                if key in axInfo:
                    v = axInfo[key]
                    v2 = self._info[ax][key]
                    shape = list(v.shape)
                    shape[0] += v2.shape[0]
                    v.resize(shape)
                    v[-v2.shape[0]:] = v2
                else:
                    raise TypeError('Cannot append to axis info key "%s"; this key is not present in the target file.' % key)
            f.close()
        else:
            f = h5py.File(fileName, 'w')
            f.attrs['MetaArray'] = MetaArray.version
            f.create_dataset('data', data=self.view(np.ndarray), **dsOpts)
            if isinstance(dsOpts['chunks'], tuple):
                dsOpts['chunks'] = True
                if 'maxshape' in dsOpts:
                    del dsOpts['maxshape']
            self.writeHDF5Meta(f, 'info', self._info, **dsOpts)
            f.close()

    def writeHDF5Meta(self, root, name, data, **dsOpts):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(data, np.ndarray):
            dsOpts['maxshape'] = (None,) + data.shape[1:]
            root.create_dataset(name, data=data, **dsOpts)
        elif isinstance(data, list) or isinstance(data, tuple):
            gr = root.create_group(name)
            if isinstance(data, list):
                gr.attrs['_metaType_'] = 'list'
            else:
                gr.attrs['_metaType_'] = 'tuple'
            for i in range(len(data)):
                self.writeHDF5Meta(gr, str(i), data[i], **dsOpts)
        elif isinstance(data, dict):
            gr = root.create_group(name)
            gr.attrs['_metaType_'] = 'dict'
            for (k, v) in data.items():
                self.writeHDF5Meta(gr, k, v, **dsOpts)
        elif isinstance(data, int) or isinstance(data, float) or isinstance(data, np.integer) or isinstance(data, np.floating):
            root.attrs[name] = data
        else:
            try:
                root.attrs[name] = repr(data)
            except:
                print("Can not store meta data of type '%s' in HDF5. (key is '%s')" % (str(type(data)), str(name)))
                raise

    def writeMa(self, fileName, appendAxis=None, newFile=False):
        if False:
            print('Hello World!')
        'Write an old-style .ma file'
        meta = {'shape': self.shape, 'type': str(self.dtype), 'info': self.infoCopy(), 'version': MetaArray.version}
        axstrs = []
        if appendAxis is not None:
            if MetaArray.isNameType(appendAxis):
                appendAxis = self._interpretAxis(appendAxis)
            ax = meta['info'][appendAxis]
            ax['values_len'] = 'dynamic'
            if 'values' in ax:
                ax['values_type'] = str(ax['values'].dtype)
                dynXVals = ax['values']
                del ax['values']
            else:
                dynXVals = None
        for ax in meta['info']:
            if 'values' in ax:
                axstrs.append(ax['values'].tostring())
                ax['values_len'] = len(axstrs[-1])
                ax['values_type'] = str(ax['values'].dtype)
                del ax['values']
        if not newFile:
            newFile = not os.path.exists(fileName) or os.stat(fileName).st_size == 0
        if appendAxis is None or newFile:
            fd = open(fileName, 'wb')
            fd.write(str(meta) + '\n\n')
            for ax in axstrs:
                fd.write(ax)
        else:
            fd = open(fileName, 'ab')
        if self.dtype != object:
            dataStr = self.view(np.ndarray).tostring()
        else:
            dataStr = pickle.dumps(self.view(np.ndarray))
        if appendAxis is not None:
            frameInfo = {'len': len(dataStr), 'numFrames': self.shape[appendAxis]}
            if dynXVals is not None:
                frameInfo['xVals'] = list(dynXVals)
            fd.write('\n' + str(frameInfo) + '\n')
        fd.write(dataStr)
        fd.close()

    def writeCsv(self, fileName=None):
        if False:
            i = 10
            return i + 15
        'Write 2D array to CSV file or return the string if no filename is given'
        if self.ndim > 2:
            raise Exception('CSV Export is only for 2D arrays')
        if fileName is not None:
            file = open(fileName, 'w')
        ret = ''
        if 'cols' in self._info[0]:
            s = ','.join([x['name'] for x in self._info[0]['cols']]) + '\n'
            if fileName is not None:
                file.write(s)
            else:
                ret += s
        for row in range(0, self.shape[1]):
            s = ','.join(['%g' % x for x in self[:, row]]) + '\n'
            if fileName is not None:
                file.write(s)
            else:
                ret += s
        if fileName is not None:
            file.close()
        else:
            return ret