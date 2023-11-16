from __future__ import annotations
import pytest
from airflow_breeze.utils.packages import get_long_package_names

@pytest.mark.parametrize('short_form_providers, expected', [pytest.param(('awesome', 'foo.bar'), ('apache-airflow-providers-awesome', 'apache-airflow-providers-foo-bar'), id='providers'), pytest.param(('apache-airflow', 'helm-chart', 'docker-stack'), ('apache-airflow', 'helm-chart', 'docker-stack'), id='non-providers-docs'), pytest.param(('apache-airflow-providers',), ('apache-airflow-providers',), id='providers-index'), pytest.param(('docker', 'docker-stack', 'apache-airflow-providers'), ('apache-airflow-providers-docker', 'docker-stack', 'apache-airflow-providers'), id='mixin')])
def test_get_provider_name_from_short_hand(short_form_providers, expected):
    if False:
        i = 10
        return i + 15
    assert get_long_package_names(short_form_providers) == expected