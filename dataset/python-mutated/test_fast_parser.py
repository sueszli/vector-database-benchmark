import re
import numpy as np
import pytest
from astropy.time import Time, TimeYearDayTime, conf
iso_times = ['2000-02-29', '1981-12-31 12:13', '1981-12-31 12:13:14', '2020-12-31 12:13:14.56']
isot_times = [re.sub(' ', 'T', tm) for tm in iso_times]
yday_times = ['2000:060', '1981:365:12:13:14', '1981:365:12:13', '2020:366:12:13:14.56']
yday_array = np.array([['2000:060', '1981:365:12:13:14'], ['1981:365:12:13', '2020:366:12:13:14.56']]).T

def test_fast_conf():
    if False:
        i = 10
        return i + 15
    assert conf.use_fast_parser == 'True'
    with pytest.raises(ValueError, match='Time 2000:0601 does not match yday format'):
        Time('2000:0601', format='yday')
    Time('2020:150:12:13:14.', format='yday')
    with conf.set_temp('use_fast_parser', 'force'):
        Time('2020:150:12:13:14.', format='yday')
    with conf.set_temp('use_fast_parser', 'False'):
        with pytest.raises(ValueError, match='could not convert string to float'):
            Time('2020:150:12:13:14.', format='yday')
    with conf.set_temp('use_fast_parser', 'False'):
        assert conf.use_fast_parser == 'False'
        with pytest.raises(ValueError, match='Time 2000:0601 does not match yday format'):
            Time('2000:0601', format='yday')
    with conf.set_temp('use_fast_parser', 'force'):
        assert conf.use_fast_parser == 'force'
        err = 'fast C time string parser failed: time string ends in middle of component'
        with pytest.raises(ValueError, match=err):
            Time('2000:0601', format='yday')

@pytest.mark.parametrize('times,format', [(iso_times, 'iso'), (isot_times, 'isot'), (yday_times, 'yday'), (yday_array, 'yday')])
@pytest.mark.parametrize('variant', [0, 1, 2])
def test_fast_matches_python(times, format, variant):
    if False:
        i = 10
        return i + 15
    if variant == 0:
        pass
    elif variant == 1:
        times = times[-1]
    elif variant == 2:
        times = [times[-1]] * 2
    with conf.set_temp('use_fast_parser', 'False'):
        tms_py = Time(times, format=format)
    with conf.set_temp('use_fast_parser', 'force'):
        tms_c = Time(times, format=format)
    assert np.all(tms_py == tms_c)

def test_fast_yday_exceptions():
    if False:
        for i in range(10):
            print('nop')
    with conf.set_temp('use_fast_parser', 'force'):
        for (times, err) in [('2020:150:12', 'time string ends at beginning of component'), ('2020:150:1', 'time string ends in middle of component'), ('2020:150*12:13:14', 'required delimiter character'), ('2020:15*:12:13:14', 'non-digit found where digit'), ('2020:999:12:13:14', 'bad day of year')]:
            with pytest.raises(ValueError, match=err):
                Time(times, format='yday')

def test_fast_iso_exceptions():
    if False:
        i = 10
        return i + 15
    with conf.set_temp('use_fast_parser', 'force'):
        for (times, err) in [('2020-10-10 12', 'time string ends at beginning of component'), ('2020-10-10 1', 'time string ends in middle of component'), ('2020*10-10 12:13:14', 'required delimiter character'), ('2020-10-10 *2:13:14', 'non-digit found where digit')]:
            with pytest.raises(ValueError, match=err):
                Time(times, format='iso')

def test_fast_non_ascii():
    if False:
        return 10
    with pytest.raises(ValueError, match='input is not pure ASCII'):
        with conf.set_temp('use_fast_parser', 'force'):
            Time('2020-01-01 1ᛦ:13:14.4324')

def test_fast_subclass():
    if False:
        i = 10
        return i + 15
    'Test subclass where use_fast_parser class attribute is not in __dict__'

    class TimeYearDayTimeSubClass(TimeYearDayTime):
        name = 'yday_subclass'
    assert hasattr(TimeYearDayTimeSubClass, 'fast_parser_pars')
    assert 'fast_parser_pars' not in TimeYearDayTimeSubClass.__dict__
    try:
        with pytest.raises(ValueError, match='Time 2000:0601 does not match yday_subclass format'):
            with conf.set_temp('use_fast_parser', 'force'):
                Time('2000:0601', format='yday_subclass')
    finally:
        del TimeYearDayTimeSubClass._registry['yday_subclass']