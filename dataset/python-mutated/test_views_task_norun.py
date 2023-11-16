from __future__ import annotations
import datetime
import urllib.parse
import pytest
from tests.test_utils.db import clear_db_runs
pytestmark = pytest.mark.db_test
DEFAULT_DATE = datetime.datetime(2022, 1, 1)
DEFAULT_VAL = urllib.parse.quote_plus(str(DEFAULT_DATE))

@pytest.fixture(scope='module', autouse=True)
def reset_dagruns():
    if False:
        print('Hello World!')
    'Clean up stray garbage from other tests.'
    clear_db_runs()

def test_task_view_no_task_instance(admin_client):
    if False:
        print('Hello World!')
    url = f'/task?task_id=runme_0&dag_id=example_bash_operator&execution_date={DEFAULT_VAL}'
    resp = admin_client.get(url, follow_redirects=True)
    assert resp.status_code == 200
    html = resp.data.decode('utf-8')
    assert '<h5>No Task Instance Available</h5>' in html
    assert '<h5>Task Instance Attributes</h5>' not in html

def test_rendered_templates_view_no_task_instance(admin_client):
    if False:
        return 10
    url = f'/rendered-templates?task_id=runme_0&dag_id=example_bash_operator&execution_date={DEFAULT_VAL}'
    resp = admin_client.get(url, follow_redirects=True)
    assert resp.status_code == 200
    html = resp.data.decode('utf-8')
    assert 'Rendered Template' in html