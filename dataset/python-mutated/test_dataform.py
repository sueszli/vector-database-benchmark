from __future__ import annotations
import pytest
from airflow.providers.google.cloud.utils.dataform import DataformLocations, define_default_location

@pytest.mark.parametrize('region, expected', [('us-central1', DataformLocations.US), ('europe-west4', DataformLocations.EUROPE)])
def test_define_default_location(region, expected):
    if False:
        return 10
    actual = define_default_location(region)
    assert actual == expected