from __future__ import annotations
import datetime
import pendulum
import pytest
from airflow.utils import timezone
from airflow.utils.timezone import coerce_datetime
CET = pendulum.tz.timezone('Europe/Paris')
EAT = pendulum.tz.timezone('Africa/Nairobi')
ICT = pendulum.tz.timezone('Asia/Bangkok')
UTC = timezone.utc

class TestTimezone:

    def test_is_aware(self):
        if False:
            i = 10
            return i + 15
        assert timezone.is_localized(datetime.datetime(2011, 9, 1, 13, 20, 30, tzinfo=EAT))
        assert not timezone.is_localized(datetime.datetime(2011, 9, 1, 13, 20, 30))

    def test_is_naive(self):
        if False:
            i = 10
            return i + 15
        assert not timezone.is_naive(datetime.datetime(2011, 9, 1, 13, 20, 30, tzinfo=EAT))
        assert timezone.is_naive(datetime.datetime(2011, 9, 1, 13, 20, 30))

    def test_utcnow(self):
        if False:
            for i in range(10):
                print('nop')
        now = timezone.utcnow()
        assert timezone.is_localized(now)
        assert now.replace(tzinfo=None) == now.astimezone(UTC).replace(tzinfo=None)

    def test_convert_to_utc(self):
        if False:
            while True:
                i = 10
        naive = datetime.datetime(2011, 9, 1, 13, 20, 30)
        utc = datetime.datetime(2011, 9, 1, 13, 20, 30, tzinfo=UTC)
        assert utc == timezone.convert_to_utc(naive)
        eat = datetime.datetime(2011, 9, 1, 13, 20, 30, tzinfo=EAT)
        utc = datetime.datetime(2011, 9, 1, 10, 20, 30, tzinfo=UTC)
        assert utc == timezone.convert_to_utc(eat)

    def test_make_naive(self):
        if False:
            return 10
        assert timezone.make_naive(datetime.datetime(2011, 9, 1, 13, 20, 30, tzinfo=EAT), EAT) == datetime.datetime(2011, 9, 1, 13, 20, 30)
        assert timezone.make_naive(datetime.datetime(2011, 9, 1, 17, 20, 30, tzinfo=ICT), EAT) == datetime.datetime(2011, 9, 1, 13, 20, 30)
        with pytest.raises(ValueError):
            timezone.make_naive(datetime.datetime(2011, 9, 1, 13, 20, 30), EAT)

    def test_make_aware(self):
        if False:
            i = 10
            return i + 15
        assert timezone.make_aware(datetime.datetime(2011, 9, 1, 13, 20, 30), EAT) == datetime.datetime(2011, 9, 1, 13, 20, 30, tzinfo=EAT)
        with pytest.raises(ValueError):
            timezone.make_aware(datetime.datetime(2011, 9, 1, 13, 20, 30, tzinfo=EAT), EAT)

    def test_td_format(self):
        if False:
            return 10
        td = datetime.timedelta(seconds=3752)
        assert timezone.td_format(td) == '1h:2M:32s'
        td = 3200.0
        assert timezone.td_format(td) == '53M:20s'
        td = 3200
        assert timezone.td_format(td) == '53M:20s'
        td = 0.123
        assert timezone.td_format(td) == '<1s'
        td = None
        assert timezone.td_format(td) is None
        td = datetime.timedelta(seconds=300752)
        assert timezone.td_format(td) == '3d:11h:32M:32s'
        td = 434343600.0
        assert timezone.td_format(td) == '13y:11m:17d:3h'

@pytest.mark.parametrize('input_datetime, output_datetime', [pytest.param(None, None, id='None datetime'), pytest.param(pendulum.DateTime(2021, 11, 1), pendulum.DateTime(2021, 11, 1, tzinfo=UTC), id='Non aware pendulum Datetime'), pytest.param(pendulum.DateTime(2021, 11, 1, tzinfo=CET), pendulum.DateTime(2021, 11, 1, tzinfo=CET), id='Aware pendulum Datetime'), pytest.param(datetime.datetime(2021, 11, 1), pendulum.DateTime(2021, 11, 1, tzinfo=UTC), id='Non aware datetime'), pytest.param(datetime.datetime(2021, 11, 1, tzinfo=CET), pendulum.DateTime(2021, 11, 1, tzinfo=CET), id='Aware datetime')])
def test_coerce_datetime(input_datetime, output_datetime):
    if False:
        while True:
            i = 10
    assert output_datetime == coerce_datetime(input_datetime)