import pandas as pd
import pytest
from ray.data.datasource.filename_provider import _DefaultFilenameProvider

@pytest.fixture(params=['csv', None])
def filename_provider(request):
    if False:
        for i in range(10):
            print('nop')
    yield _DefaultFilenameProvider(dataset_uuid='', file_format=request.param)

def test_default_filename_for_row_is_deterministic(filename_provider):
    if False:
        while True:
            i = 10
    row = {}
    first_filename = filename_provider.get_filename_for_row(row, task_index=0, block_index=0, row_index=0)
    second_filename = filename_provider.get_filename_for_row(row, task_index=0, block_index=0, row_index=0)
    assert first_filename == second_filename

def test_default_filename_for_block_is_deterministic(filename_provider):
    if False:
        i = 10
        return i + 15
    block = pd.DataFrame()
    first_filename = filename_provider.get_filename_for_block(block, task_index=0, block_index=0)
    second_filename = filename_provider.get_filename_for_block(block, task_index=0, block_index=0)
    assert first_filename == second_filename

def test_default_filename_for_row_is_unique(filename_provider):
    if False:
        return 10
    filenames = [filename_provider.get_filename_for_row({}, task_index=task_index, block_index=block_index, row_index=row_index) for task_index in range(2) for block_index in range(2) for row_index in range(2)]
    assert len(set(filenames)) == len(filenames)

def test_default_filename_for_block_is_unique(filename_provider):
    if False:
        print('Hello World!')
    filenames = [filename_provider.get_filename_for_block(pd.DataFrame(), task_index=task_index, block_index=block_index) for task_index in range(2) for block_index in range(2)]
    assert len(set(filenames)) == len(filenames)
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))