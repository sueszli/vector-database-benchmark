"""
Copyright (c) 2014-2023 Maltrail developers (https://github.com/stamparm/maltrail/)
See the file 'LICENSE' for copying permission
"""
import re

class TrailsDict(dict):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._trails = {}
        self._regex = ''
        self._infos = []
        self._reverse_infos = {}
        self._references = []
        self._reverse_references = {}

    def __delitem__(self, key):
        if False:
            print('Hello World!')
        del self._trails[key]

    def has_key(self, key):
        if False:
            i = 10
            return i + 15
        return key in self._trails

    def __contains__(self, key):
        if False:
            return 10
        return key in self._trails

    def clear(self):
        if False:
            while True:
                i = 10
        self.__init__()

    def keys(self):
        if False:
            i = 10
            return i + 15
        return self._trails.keys()

    def iterkeys(self):
        if False:
            print('Hello World!')
        for key in self._trails.keys():
            yield key

    def __iter__(self):
        if False:
            while True:
                i = 10
        for key in self._trails.keys():
            yield key

    def get(self, key, default=None):
        if False:
            for i in range(10):
                print('nop')
        if key in self._trails:
            _ = self._trails[key].split(',')
            if len(_) == 2:
                return (self._infos[int(_[0])], self._references[int(_[1])])
        return default

    def update(self, value):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(value, TrailsDict):
            if not self._trails:
                for attr in dir(self):
                    if re.search('\\A_[a-z]', attr):
                        setattr(self, attr, getattr(value, attr))
            else:
                for key in value:
                    self[key] = value[key]
        elif isinstance(value, dict):
            for key in value:
                (info, reference) = value[key]
                if info not in self._reverse_infos:
                    self._reverse_infos[info] = len(self._infos)
                    self._infos.append(info)
                if reference not in self._reverse_references:
                    self._reverse_references[reference] = len(self._references)
                    self._references.append(reference)
                self._trails[key] = '%d,%d' % (self._reverse_infos[info], self._reverse_references[reference])
        else:
            raise Exception("unsupported type '%s'" % type(value))

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self._trails)

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        if key in self._trails:
            _ = self._trails[key].split(',')
            if len(_) == 2:
                return (self._infos[int(_[0])], self._references[int(_[1])])
        raise KeyError(key)

    def __setitem__(self, key, value):
        if False:
            while True:
                i = 10
        if isinstance(value, (tuple, list)):
            (info, reference) = value
            if info not in self._reverse_infos:
                self._reverse_infos[info] = len(self._infos)
                self._infos.append(info)
            if reference not in self._reverse_references:
                self._reverse_references[reference] = len(self._references)
                self._references.append(reference)
            self._trails[key] = '%d,%d' % (self._reverse_infos[info], self._reverse_references[reference])
        else:
            raise Exception("unsupported type '%s'" % type(value))