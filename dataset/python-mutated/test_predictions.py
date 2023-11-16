from __future__ import annotations
from collections.abc import Iterator
import os
import textwrap
import conftest
import pytest
os.chdir(os.path.join('..', '..'))

@pytest.fixture(scope='session')
def test_name() -> str:
    if False:
        while True:
            i = 10
    return 'ppai/weather-predictions'

@pytest.fixture(scope='session')
def model_path_gcs(bucket_name: str) -> str:
    if False:
        print('Hello World!')
    path_gcs = f'gs://{bucket_name}/model'
    conftest.run_cmd('gsutil', 'cp', 'serving/model/*', path_gcs)
    return path_gcs

@pytest.fixture(scope='session')
def service_name(unique_name: str, location: str) -> Iterator[str]:
    if False:
        print('Hello World!')
    yield unique_name
    conftest.cloud_run_cleanup(unique_name, location)

def test_predictions(project: str, bucket_name: str, location: str, identity_token: str, service_name: str, model_path_gcs: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    conftest.run_notebook_parallel(os.path.join('notebooks', '4-predictions.ipynb'), prelude=textwrap.dedent(f'            # Google Cloud resources.\n            project = {repr(project)}\n            bucket = {repr(bucket_name)}\n            location = {repr(location)}\n            '), sections={'# 💻 Local predictions': {'variables': {'model_path_gcs': model_path_gcs}}, '# ☁️ Cloud Run predictions': {'variables': {'service_name': service_name, 'identity_token': identity_token}}})