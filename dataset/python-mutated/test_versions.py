from __future__ import annotations
import pytest
from airflow_breeze.utils.versions import strip_leading_zeros_from_version

@pytest.mark.parametrize('version,stripped_version', [('3.4.0', '3.4.0'), ('13.04.05', '13.4.5'), ('0003.00004.000005', '3.4.5')])
def test_strip_leading_versions(version: str, stripped_version):
    if False:
        while True:
            i = 10
    assert stripped_version == strip_leading_zeros_from_version(version)