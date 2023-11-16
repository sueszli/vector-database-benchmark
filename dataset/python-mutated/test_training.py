from __future__ import annotations
from datetime import datetime
import os
import tempfile
import textwrap
import conftest
import numpy as np
import pytest
from weather.data import get_inputs_patch, get_labels_patch
os.chdir(os.path.join('..', '..'))

@pytest.fixture(scope='session')
def test_name() -> str:
    if False:
        for i in range(10):
            print('nop')
    return 'ppai/weather-training'

@pytest.fixture(scope='session')
def data_path_gcs(bucket_name: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    path_gcs = f'gs://{bucket_name}/test/weather/data-training'
    date = datetime(2019, 9, 2, 18)
    point = (-69.55, -39.82)
    patch_size = 8
    inputs = get_inputs_patch(date, point, patch_size)
    labels = get_labels_patch(date, point, patch_size)
    with tempfile.NamedTemporaryFile() as f:
        batch_size = 16
        inputs_batch = [inputs] * batch_size
        labels_batch = [labels] * batch_size
        np.savez_compressed(f, inputs=inputs_batch, labels=labels_batch)
        conftest.run_cmd('gsutil', 'cp', f.name, f'{path_gcs}/example.npz')
    return path_gcs

@pytest.mark.xfail(reason='temporary API service issues')
def test_train_model(project: str, bucket_name: str, location: str, data_path_gcs: str, unique_name: str) -> None:
    if False:
        print('Hello World!')
    conftest.run_notebook_parallel(os.path.join('notebooks', '3-training.ipynb'), prelude=textwrap.dedent(f'            # Google Cloud resources.\n            project = {repr(project)}\n            bucket = {repr(bucket_name)}\n            location = {repr(location)}\n            '), sections={'# 🧠 Train the model locally': {'variables': {'data_path_gcs': data_path_gcs, 'epochs': 2}}, '# ☁️ Train the model in Vertex AI': {'variables': {'display_name': unique_name, 'data_path': data_path_gcs.replace('gs://', '/gcs/'), 'model_path': f'/gcs/{bucket_name}/test/weather/model-vertex', 'epochs': 2, 'timeout_min': 5}}})