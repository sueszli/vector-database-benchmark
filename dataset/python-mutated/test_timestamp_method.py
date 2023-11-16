from pytz import utc
from pandas._libs.tslibs import Timestamp
import pandas.util._test_decorators as td
import pandas._testing as tm

class TestTimestampMethod:

    @td.skip_if_windows
    def test_timestamp(self, fixed_now_ts):
        if False:
            print('Hello World!')
        ts = fixed_now_ts
        uts = ts.replace(tzinfo=utc)
        assert ts.timestamp() == uts.timestamp()
        tsc = Timestamp('2014-10-11 11:00:01.12345678', tz='US/Central')
        utsc = tsc.tz_convert('UTC')
        assert tsc.timestamp() == utsc.timestamp()
        with tm.set_timezone('UTC'):
            dt = ts.to_pydatetime()
            assert dt.timestamp() == ts.timestamp()