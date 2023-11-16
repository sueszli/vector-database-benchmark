import os
import pytest
from .get_client_id import get_client_id
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
COMPOSER_LOCATION = os.environ['COMPOSER_LOCATION']
COMPOSER_ENVIRONMENT = os.environ['COMPOSER_ENVIRONMENT']
COMPOSER2_ENVIRONMENT = os.environ['COMPOSER2_ENVIRONMENT']

def test_get_client_id(capsys):
    if False:
        print('Hello World!')
    get_client_id(PROJECT, COMPOSER_LOCATION, COMPOSER_ENVIRONMENT)
    (out, _) = capsys.readouterr()
    assert out.endswith('.apps.googleusercontent.com\n') is True

def test_get_client_id_composer_2(capsys):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(RuntimeError):
        get_client_id(PROJECT, COMPOSER_LOCATION, COMPOSER2_ENVIRONMENT)
        (out, _) = capsys.readouterr()
        assert 'This script is intended to be used with Composer 1' in out