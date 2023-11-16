__author__ = 'maartenbreddels'
import os
import mmap
import itertools
import functools
import collections
import logging
import numpy as np
import numpy.ma
import vaex
import astropy.table
import astropy.units
from vaex.utils import ensure_string
import re
import six
from vaex.dataset import DatasetArrays, DatasetFile
logger = logging.getLogger('vaex.file')
import vaex.dataset
import vaex.file
dataset_type_map = {}
from vaex.expression import Expression
from vaex.dataset_mmap import DatasetMemoryMapped
import struct

class HansMemoryMapped(DatasetMemoryMapped):

    def __init__(self, filename, filename_extra=None):
        if False:
            for i in range(10):
                print('nop')
        super(HansMemoryMapped, self).__init__(filename)
        (self.pageSize, self.formatSize, self.numberParticles, self.numberTimes, self.numberParameters, self.numberCompute, self.dataOffset, self.dataHeaderSize) = struct.unpack('Q' * 8, self.mapping[:8 * 8])
        zerooffset = offset = self.dataOffset
        length = self.numberParticles + 1
        stride = self.formatSize // 8
        lastoffset = offset + (self.numberParticles + 1) * (self.numberTimes - 2) * self.formatSize
        t_index = 3
        names = 'x y z vx vy vz'.split()
        midoffset = offset + (self.numberParticles + 1) * self.formatSize * t_index
        names = 'x y z vx vy vz'.split()
        for (i, name) in enumerate(names):
            self.addColumn(name + '_0', offset + 8 * i, length, dtype=np.float64, stride=stride)
        for (i, name) in enumerate(names):
            self.addColumn(name + '_last', lastoffset + 8 * i, length, dtype=np.float64, stride=stride)
        names = 'x y z vx vy vz'.split()
        if 1:
            stride = self.formatSize // 8
            for (i, name) in enumerate(names):
                self.addRank1(name, offset + 8 * i, length=self.numberParticles + 1, length1=self.numberTimes - 1, dtype=np.float64, stride=stride, stride1=1)
        if filename_extra is None:
            basename = os.path.basename(filename)
            if os.path.exists(basename + '.omega2'):
                filename_extra = basename + '.omega2'
        if filename_extra is not None:
            self.addFile(filename_extra)
            mapping = self.mapping_map[filename_extra]
            names = 'J_r J_theta J_phi Theta_r Theta_theta Theta_phi Omega_r Omega_theta Omega_phi r_apo r_peri'.split()
            offset = 0
            stride = 11
            for (i, name) in enumerate(names):
                self.addRank1(name, offset + 8 * i, length=self.numberParticles + 1, length1=self.numberTimes - 1, dtype=np.float64, stride=stride, stride1=1, filename=filename_extra)
                self.addColumn(name + '_0', offset + 8 * i, length, dtype=np.float64, stride=stride, filename=filename_extra)
                self.addColumn(name + '_last', offset + 8 * i + (self.numberParticles + 1) * (self.numberTimes - 2) * 11 * 8, length, dtype=np.float64, stride=stride, filename=filename_extra)

    @classmethod
    def can_open(cls, path, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return os.path.splitext(path)[-1] == '.bin'
        (basename, ext) = os.path.splitext(path)

    @classmethod
    def get_options(cls, path):
        if False:
            print('Hello World!')
        return []

    @classmethod
    def option_to_args(cls, option):
        if False:
            while True:
                i = 10
        return []
dataset_type_map['buist'] = HansMemoryMapped

def _python_save_name(name, used=[]):
    if False:
        while True:
            i = 10
    (first, rest) = (name[0], name[1:])
    name = re.sub('[^a-zA-Z_]', '_', first) + re.sub('[^a-zA-Z_0-9]', '_', rest)
    if name in used:
        nr = 1
        while name + '_%d' % nr in used:
            nr += 1
        name = name + '_%d' % nr
    return name

def _try_unit(unit):
    if False:
        while True:
            i = 10
    try:
        unit = astropy.units.Unit(str(unit))
        if not isinstance(unit, astropy.units.UnrecognizedUnit):
            return unit
    except:
        pass
    try:
        unit_mangle = re.match('.*\\[(.*)\\]', str(unit)).groups()[0]
        unit = astropy.units.Unit(unit_mangle)
    except:
        pass
    if isinstance(unit, six.string_types):
        return None
    elif isinstance(unit, astropy.units.UnrecognizedUnit):
        return None
    else:
        return unit

class SoneiraPeebles(DatasetArrays):

    def __init__(self, dimension, eta, max_level, L):
        if False:
            while True:
                i = 10
        columns = {}

        def todim(value):
            if False:
                while True:
                    i = 10
            if isinstance(value, (tuple, list)):
                assert len(value) >= dimension, 'either a scalar or sequence of length equal to or larger than the dimension'
                return value[:dimension]
            else:
                return [value] * dimension
        eta = eta
        max_level = max_level
        N = eta ** max_level
        array = np.zeros((dimension + 1, N), dtype=np.float64)
        L = todim(L)
        for d in range(dimension):
            vaex.vaexfast.soneira_peebles(array[d], 0, 1, L[d], eta, max_level)
        for (d, name) in zip(list(range(dimension)), 'x y z w v u'.split()):
            columns[name] = array[d]
        super(SoneiraPeebles, self).__init__(columns)

class Zeldovich(DatasetArrays):

    def __init__(self, dim=2, N=256, n=-2.5, t=None, seed=None, scale=1, name='zeldovich approximation'):
        if False:
            while True:
                i = 10
        super(Zeldovich, self).__init__(name=name)
        if seed is not None:
            np.random.seed(seed)
        shape = (N,) * dim
        A = np.random.normal(0.0, 1.0, shape)
        F = np.fft.fftn(A)
        K = np.fft.fftfreq(N, 1.0 / (2 * np.pi))[np.indices(shape)]
        k = (K ** 2).sum(axis=0)
        k_max = np.pi
        F *= np.where(np.sqrt(k) > k_max, 0, np.sqrt(k ** n) * np.exp(-k * 4.0))
        F.flat[0] = 0
        grf = np.fft.ifftn(F).real
        Q = np.indices(shape) / float(N - 1) - 0.5
        s = np.array(np.gradient(grf)) / float(N)
        s /= s.max() * 100.0
        t = t or 1.0
        X = Q + s * t
        for (d, name) in zip(list(range(dim)), 'xyzw'):
            self.add_column(name, X[d].reshape(-1) * scale)
        for (d, name) in zip(list(range(dim)), 'xyzw'):
            self.add_column('v' + name, s[d].reshape(-1) * scale)
        for (d, name) in zip(list(range(dim)), 'xyzw'):
            self.add_column(name + '0', Q[d].reshape(-1) * scale)
        return

class AsciiTable(DatasetMemoryMapped):

    def __init__(self, filename):
        if False:
            i = 10
            return i + 15
        super(AsciiTable, self).__init__(filename, nommap=True)
        import asciitable
        table = asciitable.read(filename)
        logger.debug('done parsing ascii table')
        names = table.dtype.names
        for i in range(len(table.dtype)):
            name = table.dtype.names[i]
            type = table.dtype[i]
            if type.kind in ['f', 'i']:
                self.addColumn(name, array=table[name])

    @classmethod
    def can_open(cls, path, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        can_open = path.endswith('.asc')
        logger.debug('%r can open: %r' % (cls.__name__, can_open))
        return can_open