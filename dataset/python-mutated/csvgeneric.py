from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime
import itertools
from .. import feed, TimeFrame
from ..utils import date2num
from ..utils.py3 import integer_types, string_types

class GenericCSVData(feed.CSVDataBase):
    """Parses a CSV file according to the order and field presence defined by the
    parameters

    Specific parameters (or specific meaning):

      - ``dataname``: The filename to parse or a file-like object

      - The lines parameters (datetime, open, high ...) take numeric values

        A value of -1 indicates absence of that field in the CSV source

      - If ``time`` is present (parameter time >=0) the source contains
        separated fields for date and time, which will be combined

      - ``nullvalue``

        Value that will be used if a value which should be there is missing
        (the CSV field is empty)

      - ``dtformat``: Format used to parse the datetime CSV field. See the
        python strptime/strftime documentation for the format.

        If a numeric value is specified, it will be interpreted as follows

          - ``1``: The value is a Unix timestamp of type ``int`` representing
            the number of seconds since Jan 1st, 1970

          - ``2``: The value is a Unix timestamp of type ``float``

        If a **callable** is passed

          - it will accept a string and return a `datetime.datetime` python
            instance

      - ``tmformat``: Format used to parse the time CSV field if "present"
        (the default for the "time" CSV field is not to be present)

    """
    params = (('nullvalue', float('NaN')), ('dtformat', '%Y-%m-%d %H:%M:%S'), ('tmformat', '%H:%M:%S'), ('datetime', 0), ('time', -1), ('open', 1), ('high', 2), ('low', 3), ('close', 4), ('volume', 5), ('openinterest', 6))

    def start(self):
        if False:
            while True:
                i = 10
        super(GenericCSVData, self).start()
        self._dtstr = False
        if isinstance(self.p.dtformat, string_types):
            self._dtstr = True
        elif isinstance(self.p.dtformat, integer_types):
            idt = int(self.p.dtformat)
            if idt == 1:
                self._dtconvert = lambda x: datetime.utcfromtimestamp(int(x))
            elif idt == 2:
                self._dtconvert = lambda x: datetime.utcfromtimestamp(float(x))
        else:
            self._dtconvert = self.p.dtformat

    def _loadline(self, linetokens):
        if False:
            while True:
                i = 10
        dtfield = linetokens[self.p.datetime]
        if self._dtstr:
            dtformat = self.p.dtformat
            if self.p.time >= 0:
                dtfield += 'T' + linetokens[self.p.time]
                dtformat += 'T' + self.p.tmformat
            dt = datetime.strptime(dtfield, dtformat)
        else:
            dt = self._dtconvert(dtfield)
        if self.p.timeframe >= TimeFrame.Days:
            if self._tzinput:
                dtin = self._tzinput.localize(dt)
            else:
                dtin = dt
            dtnum = date2num(dtin)
            dteos = datetime.combine(dt.date(), self.p.sessionend)
            dteosnum = self.date2num(dteos)
            if dteosnum > dtnum:
                self.lines.datetime[0] = dteosnum
            else:
                self.l.datetime[0] = date2num(dt) if self._tzinput else dtnum
        else:
            self.lines.datetime[0] = date2num(dt)
        for linefield in (x for x in self.getlinealiases() if x != 'datetime'):
            csvidx = getattr(self.params, linefield)
            if csvidx is None or csvidx < 0:
                csvfield = self.p.nullvalue
            else:
                csvfield = linetokens[csvidx]
            if csvfield == '':
                csvfield = self.p.nullvalue
            line = getattr(self.lines, linefield)
            line[0] = float(float(csvfield))
        return True

class GenericCSV(feed.CSVFeedBase):
    DataCls = GenericCSVData