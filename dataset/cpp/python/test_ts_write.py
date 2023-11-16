from datetime import datetime as dt

import pytest
import pytz
from pandas.testing import assert_frame_equal
from tests.util import assert_frame_equal_

from arctic.date import mktz, DateRange
from arctic.exceptions import OverlappingDataException

DUMMY_DATA = [
              {'a': 1.,
               'b': 2.,
               'index': dt(2013, 1, 1, tzinfo=mktz('Europe/London'))
               },
              {'b': 3.,
               'c': 4.,
               'index': dt(2013, 1, 2, tzinfo=mktz('Europe/London'))
               },
              {'b': 5.,
               'c': 6.,
               'index': dt(2013, 1, 3, tzinfo=mktz('Europe/London'))
               },
              {'b': 7.,
               'c': 8.,
               'index': dt(2013, 1, 4, tzinfo=mktz('Europe/London'))
               },
              {'b': 9.,
               'c': 10.,
               'index': dt(2013, 7, 5, tzinfo=mktz('Europe/London'))
               },
              ]


def test_ts_write_simple(tickstore_lib):
    assert tickstore_lib.stats()['chunks']['count'] == 0
    tickstore_lib.write('SYM', DUMMY_DATA)
    assert tickstore_lib.stats()['chunks']['count'] == 1
    assert len(tickstore_lib.read('SYM')) == 5
    assert tickstore_lib.list_symbols() == ['SYM']


def test_overlapping_load(tickstore_lib):
    data = DUMMY_DATA
    tickstore_lib.write('SYM', DUMMY_DATA)
    with pytest.raises(OverlappingDataException):
        tickstore_lib.write('SYM', data)

    data = DUMMY_DATA[2:]
    with pytest.raises(OverlappingDataException):
        tickstore_lib.write('SYM', data)

    data = DUMMY_DATA[2:3]
    with pytest.raises(OverlappingDataException):
        tickstore_lib.write('SYM', data)

    # overlapping at the beginning is ok
    data = [DUMMY_DATA[0]]
    tickstore_lib.write('SYM', data)

    # overlapping at the end is ok
    data = [DUMMY_DATA[-1]]
    tickstore_lib.write('SYM', data)


def test_ts_write_pandas(tickstore_lib):
    data = DUMMY_DATA
    tickstore_lib.write('SYM', data)

    data = tickstore_lib.read('SYM', columns=None)
    assert data.index[0] == dt(2013, 1, 1, tzinfo=mktz('Europe/London'))
    assert data.a[0] == 1
    tickstore_lib.delete('SYM')
    tickstore_lib.write('SYM', data)

    read = tickstore_lib.read('SYM', columns=None)
    assert_frame_equal_(read, data, check_names=False)


def test_ts_write_named_col(tickstore_lib):
    data = DUMMY_DATA
    tickstore_lib.write('SYM', data)

    data = tickstore_lib.read('SYM')
    assert data.index[0] == dt(2013, 1, 1, tzinfo=mktz('Europe/London'))
    assert data.a[0] == 1
    assert(data.index.name is None)
    data.index.name = 'IndexName'

    tickstore_lib.delete('SYM')
    tickstore_lib.write('SYM', data)

    read = tickstore_lib.read('SYM')
    assert(read.index.name is None)


def test_millisecond_roundtrip(tickstore_lib):
    test_time = dt(2004, 1, 14, 8, 30, 4, 807000, tzinfo=pytz.utc)

    data = [{'index': test_time, 'price': 9142.12, 'qualifiers': ''}]
    tickstore_lib.write('blah', data)

    data_range = DateRange(dt(2004, 1, 14, tzinfo=pytz.utc),
                           dt(2004, 1, 15, tzinfo=pytz.utc))
    reread = tickstore_lib.read('blah', data_range)

    assert reread.index[0].to_pydatetime() == test_time
