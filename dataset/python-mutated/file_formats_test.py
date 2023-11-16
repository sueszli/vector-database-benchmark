from pathlib import Path
import pytest
from airbyte_cdk import AirbyteLogger
from airbyte_cdk.utils import AirbyteTracedException
from source_file import SourceFile
from source_file.client import Client
SAMPLE_DIRECTORY = Path(__file__).resolve().parent.joinpath('sample_files/formats')

def check_read(config, expected_columns=10, expected_rows=42):
    if False:
        while True:
            i = 10
    client = Client(**config)
    rows = list(client.read())
    assert len(rows) == expected_rows
    assert len(rows[0]) == expected_columns

@pytest.mark.parametrize('file_format, extension, expected_columns, expected_rows, filename', [('csv', 'csv', 8, 5000, 'demo'), ('json', 'json', 2, 1, 'demo'), ('jsonl', 'jsonl', 2, 10, 'jsonl_nested'), ('jsonl', 'jsonl', 2, 6492, 'jsonl'), ('excel', 'xls', 8, 50, 'demo'), ('excel', 'xlsx', 8, 50, 'demo'), ('feather', 'feather', 9, 3, 'demo'), ('parquet', 'parquet', 9, 3, 'demo'), ('yaml', 'yaml', 8, 3, 'demo')])
def test_local_file_read(file_format, extension, expected_columns, expected_rows, filename):
    if False:
        return 10
    file_directory = SAMPLE_DIRECTORY.joinpath(file_format)
    file_path = str(file_directory.joinpath(f'{filename}.{extension}'))
    configs = {'dataset_name': 'test', 'format': file_format, 'url': file_path, 'provider': {'storage': 'local'}}
    check_read(configs, expected_columns, expected_rows)

@pytest.mark.parametrize('file_format, extension, wrong_format, filename', [('excel', 'xls', 'csv', 'demo'), ('excel', 'xlsx', 'csv', 'demo'), ('csv', 'csv', 'excel', 'demo'), ('csv', 'csv', 'excel', 'demo'), ('jsonl', 'jsonl', 'excel', 'jsonl_nested'), ('feather', 'feather', 'csv', 'demo'), ('parquet', 'parquet', 'feather', 'demo'), ('yaml', 'yaml', 'json', 'demo')])
def test_raises_file_wrong_format(file_format, extension, wrong_format, filename):
    if False:
        while True:
            i = 10
    file_directory = SAMPLE_DIRECTORY.joinpath(file_format)
    file_path = str(file_directory.joinpath(f'{filename}.{extension}'))
    configs = {'dataset_name': 'test', 'format': wrong_format, 'url': file_path, 'provider': {'storage': 'local'}}
    client = Client(**configs)
    with pytest.raises((TypeError, ValueError, AirbyteTracedException)):
        list(client.read())

def run_load_dataframes(config, expected_columns=10, expected_rows=42):
    if False:
        print('Hello World!')
    df_list = SourceFile.load_dataframes(config=config, logger=AirbyteLogger(), skip_data=False)
    assert len(df_list) == 1
    df = df_list[0]
    assert len(df.columns) == expected_columns
    assert len(df.index) == expected_rows
    return df

def run_load_nested_json_schema(config, expected_columns=10, expected_rows=42):
    if False:
        for i in range(10):
            print('nop')
    data_list = SourceFile.load_nested_json(config, logger=AirbyteLogger())
    assert len(data_list) == 1
    df = data_list[0]
    assert len(df) == expected_rows
    return df