from datetime import datetime
from pytest import fixture
from source_rki_covid.source import StatesHistoryRecovered

@fixture
def patch_states_history_recovered(mocker):
    if False:
        i = 10
        return i + 15
    mocker.patch.object(StatesHistoryRecovered, 'primary_key', None)

def check_diff(start_date):
    if False:
        print('Hello World!')
    diff = datetime.now() - datetime.strptime(start_date, '%Y-%m-%d')
    if diff.days <= 0:
        return str(1)
    return str(diff.days)

def test_parse_with_cases(patch_states_history_recovered):
    if False:
        i = 10
        return i + 15
    config = {'start_date': '2022-04-27'}
    stream = StatesHistoryRecovered(config)
    expected_stream_path = 'states/history/recovered/' + check_diff(config.get('start_date'))
    assert stream.path() == expected_stream_path