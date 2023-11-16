import os
from .get_dag_prefix import get_dag_prefix
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
COMPOSER_LOCATION = os.environ['COMPOSER_LOCATION']
COMPOSER_ENVIRONMENT = os.environ['COMPOSER_ENVIRONMENT']

def test_get_dag_prefix(capsys):
    if False:
        print('Hello World!')
    get_dag_prefix(PROJECT, COMPOSER_LOCATION, COMPOSER_ENVIRONMENT)
    (out, _) = capsys.readouterr()
    assert 'gs://' in out