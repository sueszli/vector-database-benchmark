from datetime import datetime as dt
import pandas as pd
import pytest
from pandas import DataFrame, MultiIndex
from pandas.testing import assert_frame_equal
from arctic.chunkstore.date_chunker import DateChunker
from arctic.date import DateRange

def test_date_filter():
    if False:
        print('Hello World!')
    c = DateChunker()
    df = DataFrame(data={'data': [1, 2, 3]}, index=MultiIndex.from_tuples([(dt(2016, 1, 1), 1), (dt(2016, 1, 2), 1), (dt(2016, 1, 3), 1)], names=['date', 'id']))
    assert_frame_equal(c.filter(df, DateRange(None, dt(2016, 1, 3))), df)
    assert_frame_equal(c.filter(df, DateRange(dt(2016, 1, 1), None)), df)
    assert_frame_equal(c.filter(df, DateRange(None, None)), df)
    assert_frame_equal(c.filter(df, DateRange(dt(2000, 1, 1), None)), df)
    assert c.filter(df, DateRange(dt(2020, 1, 2), None)).empty
    assert_frame_equal(c.filter(df, DateRange(None, dt(2020, 1, 1))), df)
    assert c.filter(df, DateRange(dt(2017, 1, 1), dt(2018, 1, 1))).empty

def test_date_filter_no_index():
    if False:
        for i in range(10):
            print('nop')
    c = DateChunker()
    df = DataFrame(data={'data': [1, 2, 3], 'date': [dt(2016, 1, 1), dt(2016, 1, 2), dt(2016, 1, 3)]})
    assert_frame_equal(c.filter(df, DateRange(None, dt(2016, 1, 3))), df)
    assert_frame_equal(c.filter(df, DateRange(dt(2016, 1, 1), None)), df)
    assert_frame_equal(c.filter(df, DateRange(None, None)), df)
    assert_frame_equal(c.filter(df, DateRange(dt(2000, 1, 1), None)), df)
    assert c.filter(df, DateRange(dt(2020, 1, 2), None)).empty
    assert_frame_equal(c.filter(df, DateRange(None, dt(2020, 1, 1))), df)
    assert c.filter(df, DateRange(dt(2017, 1, 1), dt(2018, 1, 1))).empty

def test_date_filter_with_pd_date_range():
    if False:
        return 10
    c = DateChunker()
    df = DataFrame(data={'data': [1, 2, 3]}, index=MultiIndex.from_tuples([(dt(2016, 1, 1), 1), (dt(2016, 1, 2), 1), (dt(2016, 1, 3), 1)], names=['date', 'id']))
    assert c.filter(df, pd.date_range(dt(2017, 1, 1), dt(2018, 1, 1))).empty
    assert_frame_equal(c.filter(df, pd.date_range(dt(2016, 1, 1), dt(2017, 1, 1))), df)

def test_to_chunks_exceptions():
    if False:
        i = 10
        return i + 15
    df = DataFrame(data={'data': [1, 2, 3]})
    c = DateChunker()
    with pytest.raises(Exception) as e:
        next(c.to_chunks(df, 'D'))
    assert 'datetime indexed' in str(e.value)
    df.columns = ['date']
    with pytest.raises(Exception) as e:
        next(c.to_chunks(df, 'ZSDFG'))
    assert 'Unknown freqstr' in str(e.value) or 'Invalid frequency' in str(e.value)

def test_exclude():
    if False:
        return 10
    c = DateChunker()
    df = DataFrame(data={'data': [1, 2, 3]}, index=MultiIndex.from_tuples([(dt(2016, 1, 1), 1), (dt(2016, 1, 2), 1), (dt(2016, 1, 3), 1)], names=['date', 'id']))
    df2 = DataFrame(data={'data': [1, 2, 3]})
    assert c.exclude(df, DateRange(dt(2016, 1, 1), dt(2016, 1, 1))).equals(c.exclude(df, pd.date_range(dt(2016, 1, 1), dt(2016, 1, 1))))
    assert c.exclude(df2, None).equals(df2)

def test_exclude_no_index():
    if False:
        i = 10
        return i + 15
    c = DateChunker()
    df = DataFrame(data={'data': [1, 2, 3], 'date': [dt(2016, 1, 1), dt(2016, 1, 2), dt(2016, 1, 3)]})
    df2 = DataFrame(data={'data': [1, 2, 3]})
    assert c.exclude(df, DateRange(dt(2016, 1, 1), dt(2016, 1, 1))).equals(c.exclude(df, pd.date_range(dt(2016, 1, 1), dt(2016, 1, 1))))
    assert c.exclude(df2, None).equals(df2)

def test_with_tuples():
    if False:
        for i in range(10):
            print('nop')
    c = DateChunker()
    df = DataFrame(data={'data': [1, 2, 3], 'date': [dt(2016, 1, 1), dt(2016, 1, 2), dt(2016, 1, 3)]})
    assert_frame_equal(c.filter(df, (None, dt(2016, 1, 3))), df)
    assert_frame_equal(c.filter(df, (dt(2016, 1, 1), None)), df)
    assert_frame_equal(c.filter(df, (None, None)), df)
    assert_frame_equal(c.filter(df, (dt(2000, 1, 1), None)), df)
    assert c.filter(df, (dt(2020, 1, 2), None)).empty
    assert_frame_equal(c.filter(df, (None, dt(2020, 1, 1))), df)
    assert c.filter(df, (dt(2017, 1, 1), dt(2018, 1, 1))).empty