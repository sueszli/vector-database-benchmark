from __future__ import absolute_import, division, print_function, unicode_literals
from backtrader.utils.py3 import filter, string_types, integer_types
from backtrader import date2num
import backtrader.feed as feed

class PandasDirectData(feed.DataBase):
    """
    Uses a Pandas DataFrame as the feed source, iterating directly over the
    tuples returned by "itertuples".

    This means that all parameters related to lines must have numeric
    values as indices into the tuples

    Note:

      - The ``dataname`` parameter is a Pandas DataFrame

      - A negative value in any of the parameters for the Data lines
        indicates it's not present in the DataFrame
        it is
    """
    params = (('datetime', 0), ('open', 1), ('high', 2), ('low', 3), ('close', 4), ('volume', 5), ('openinterest', 6))
    datafields = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest']

    def start(self):
        if False:
            return 10
        super(PandasDirectData, self).start()
        self._rows = self.p.dataname.itertuples()

    def _load(self):
        if False:
            i = 10
            return i + 15
        try:
            row = next(self._rows)
        except StopIteration:
            return False
        for datafield in self.getlinealiases():
            if datafield == 'datetime':
                continue
            colidx = getattr(self.params, datafield)
            if colidx < 0:
                continue
            line = getattr(self.lines, datafield)
            line[0] = row[colidx]
        colidx = getattr(self.params, 'datetime')
        tstamp = row[colidx]
        dt = tstamp.to_pydatetime()
        dtnum = date2num(dt)
        line = getattr(self.lines, 'datetime')
        line[0] = dtnum
        return True

class PandasData(feed.DataBase):
    """
    Uses a Pandas DataFrame as the feed source, using indices into column
    names (which can be "numeric")

    This means that all parameters related to lines must have numeric
    values as indices into the tuples

    Params:

      - ``nocase`` (default *True*) case insensitive match of column names

    Note:

      - The ``dataname`` parameter is a Pandas DataFrame

      - Values possible for datetime

        - None: the index contains the datetime
        - -1: no index, autodetect column
        - >= 0 or string: specific colum identifier

      - For other lines parameters

        - None: column not present
        - -1: autodetect
        - >= 0 or string: specific colum identifier
    """
    params = (('nocase', True), ('datetime', None), ('open', -1), ('high', -1), ('low', -1), ('close', -1), ('volume', -1), ('openinterest', -1))
    datafields = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest']

    def __init__(self):
        if False:
            while True:
                i = 10
        super(PandasData, self).__init__()
        colnames = list(self.p.dataname.columns.values)
        if self.p.datetime is None:
            pass
        cstrings = filter(lambda x: isinstance(x, string_types), colnames)
        colsnumeric = not len(list(cstrings))
        self._colmapping = dict()
        for datafield in self.getlinealiases():
            defmapping = getattr(self.params, datafield)
            if isinstance(defmapping, integer_types) and defmapping < 0:
                for colname in colnames:
                    if isinstance(colname, string_types):
                        if self.p.nocase:
                            found = datafield.lower() == colname.lower()
                        else:
                            found = datafield == colname
                        if found:
                            self._colmapping[datafield] = colname
                            break
                if datafield not in self._colmapping:
                    self._colmapping[datafield] = None
                    continue
            else:
                self._colmapping[datafield] = defmapping

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        super(PandasData, self).start()
        self._idx = -1
        if self.p.nocase:
            colnames = [x.lower() for x in self.p.dataname.columns.values]
        else:
            colnames = [x for x in self.p.dataname.columns.values]
        for (k, v) in self._colmapping.items():
            if v is None:
                continue
            if isinstance(v, string_types):
                try:
                    if self.p.nocase:
                        v = colnames.index(v.lower())
                    else:
                        v = colnames.index(v)
                except ValueError as e:
                    defmap = getattr(self.params, k)
                    if isinstance(defmap, integer_types) and defmap < 0:
                        v = None
                    else:
                        raise e
            self._colmapping[k] = v

    def _load(self):
        if False:
            i = 10
            return i + 15
        self._idx += 1
        if self._idx >= len(self.p.dataname):
            return False
        for datafield in self.getlinealiases():
            if datafield == 'datetime':
                continue
            colindex = self._colmapping[datafield]
            if colindex is None:
                continue
            line = getattr(self.lines, datafield)
            line[0] = self.p.dataname.iloc[self._idx, colindex]
        coldtime = self._colmapping['datetime']
        if coldtime is None:
            tstamp = self.p.dataname.index[self._idx]
        else:
            tstamp = self.p.dataname.iloc[self._idx, coldtime]
        dt = tstamp.to_pydatetime()
        dtnum = date2num(dt)
        self.lines.datetime[0] = dtnum
        return True