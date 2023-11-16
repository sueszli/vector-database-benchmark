import os
import backoff
from google.api_core.exceptions import InternalServerError
import pytest
import list_operation_status
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']

@pytest.mark.slow
def test_list_operation_status(capsys):
    if False:
        for i in range(10):
            print('nop')

    @backoff.on_exception(backoff.expo, InternalServerError, max_time=120)
    def run_sample():
        if False:
            i = 10
            return i + 15
        list_operation_status.list_operation_status(PROJECT_ID)
    run_sample()
    (out, _) = capsys.readouterr()
    assert 'Operation details' in out