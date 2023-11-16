from datetime import datetime
from pytest import fixture
from source_rki_covid.source import StatesHistoryIncidence

@fixture
def patch_states_history_incidence(mocker):
    if False:
        return 10
    mocker.patch.object(StatesHistoryIncidence, 'primary_key', None)

def check_diff(start_date):
    if False:
        i = 10
        return i + 15
    diff = datetime.now() - datetime.strptime(start_date, '%Y-%m-%d')
    if diff.days <= 0:
        return str(1)
    return str(diff.days)

def test_parse_with_cases(patch_states_history_incidence):
    if False:
        return 10
    config = {'start_date': '2022-04-27'}
    stream = StatesHistoryIncidence(config)
    expected_stream_path = 'states/history/incidence/' + check_diff(config.get('start_date'))
    assert stream.path() == expected_stream_path