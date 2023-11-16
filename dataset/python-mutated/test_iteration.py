class TestIteration:

    def test_keys(self, datetime_series):
        if False:
            for i in range(10):
                print('nop')
        assert datetime_series.keys() is datetime_series.index

    def test_iter_datetimes(self, datetime_series):
        if False:
            return 10
        for (i, val) in enumerate(datetime_series):
            assert val == datetime_series.iloc[i]

    def test_iter_strings(self, string_series):
        if False:
            i = 10
            return i + 15
        for (i, val) in enumerate(string_series):
            assert val == string_series.iloc[i]

    def test_iteritems_datetimes(self, datetime_series):
        if False:
            return 10
        for (idx, val) in datetime_series.items():
            assert val == datetime_series[idx]

    def test_iteritems_strings(self, string_series):
        if False:
            for i in range(10):
                print('nop')
        for (idx, val) in string_series.items():
            assert val == string_series[idx]
        assert not hasattr(string_series.items(), 'reverse')

    def test_items_datetimes(self, datetime_series):
        if False:
            while True:
                i = 10
        for (idx, val) in datetime_series.items():
            assert val == datetime_series[idx]

    def test_items_strings(self, string_series):
        if False:
            return 10
        for (idx, val) in string_series.items():
            assert val == string_series[idx]
        assert not hasattr(string_series.items(), 'reverse')