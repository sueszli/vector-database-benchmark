def test_chrono_system_clock():
    if False:
        i = 10
        return i + 15
    from pybind11_tests import test_chrono1
    import datetime
    date1 = test_chrono1()
    date2 = datetime.datetime.today()
    assert isinstance(date1, datetime.datetime)
    diff = abs(date1 - date2)
    assert diff.days == 0
    assert diff.seconds == 0
    assert diff.microseconds < 500000

def test_chrono_system_clock_roundtrip():
    if False:
        return 10
    from pybind11_tests import test_chrono2
    import datetime
    date1 = datetime.datetime.today()
    date2 = test_chrono2(date1)
    assert isinstance(date2, datetime.datetime)
    diff = abs(date1 - date2)
    assert diff.days == 0
    assert diff.seconds == 0
    assert diff.microseconds == 0

def test_chrono_duration_roundtrip():
    if False:
        while True:
            i = 10
    from pybind11_tests import test_chrono3
    import datetime
    date1 = datetime.datetime.today()
    date2 = datetime.datetime.today()
    diff = date2 - date1
    assert isinstance(diff, datetime.timedelta)
    cpp_diff = test_chrono3(diff)
    assert cpp_diff.days == diff.days
    assert cpp_diff.seconds == diff.seconds
    assert cpp_diff.microseconds == diff.microseconds

def test_chrono_duration_subtraction_equivalence():
    if False:
        for i in range(10):
            print('nop')
    from pybind11_tests import test_chrono4
    import datetime
    date1 = datetime.datetime.today()
    date2 = datetime.datetime.today()
    diff = date2 - date1
    cpp_diff = test_chrono4(date2, date1)
    assert cpp_diff.days == diff.days
    assert cpp_diff.seconds == diff.seconds
    assert cpp_diff.microseconds == diff.microseconds

def test_chrono_steady_clock():
    if False:
        i = 10
        return i + 15
    from pybind11_tests import test_chrono5
    import datetime
    time1 = test_chrono5()
    time2 = test_chrono5()
    assert isinstance(time1, datetime.timedelta)
    assert isinstance(time2, datetime.timedelta)

def test_chrono_steady_clock_roundtrip():
    if False:
        i = 10
        return i + 15
    from pybind11_tests import test_chrono6
    import datetime
    time1 = datetime.timedelta(days=10, seconds=10, microseconds=100)
    time2 = test_chrono6(time1)
    assert isinstance(time2, datetime.timedelta)
    assert time1.days == time2.days
    assert time1.seconds == time2.seconds
    assert time1.microseconds == time2.microseconds

def test_floating_point_duration():
    if False:
        while True:
            i = 10
    from pybind11_tests import test_chrono7
    import datetime
    time = test_chrono7(35.525123)
    assert isinstance(time, datetime.timedelta)
    assert time.seconds == 35
    assert 525122 <= time.microseconds <= 525123