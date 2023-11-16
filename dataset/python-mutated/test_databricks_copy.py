from __future__ import annotations
import pytest
from airflow.exceptions import AirflowException
from airflow.providers.databricks.operators.databricks_sql import DatabricksCopyIntoOperator
DATE = '2017-04-20'
TASK_ID = 'databricks-sql-operator'
DEFAULT_CONN_ID = 'databricks_default'
COPY_FILE_LOCATION = 's3://my-bucket/jsonData'

def test_copy_with_files():
    if False:
        while True:
            i = 10
    op = DatabricksCopyIntoOperator(file_location=COPY_FILE_LOCATION, file_format='JSON', table_name='test', files=['file1', 'file2', 'file3'], format_options={'dateFormat': 'yyyy-MM-dd'}, task_id=TASK_ID)
    assert op._create_sql_query() == f"COPY INTO test\nFROM '{COPY_FILE_LOCATION}'\nFILEFORMAT = JSON\nFILES = ('file1','file2','file3')\nFORMAT_OPTIONS ('dateFormat' = 'yyyy-MM-dd')\n".strip()

def test_copy_with_expression():
    if False:
        while True:
            i = 10
    expression = 'col1, col2'
    op = DatabricksCopyIntoOperator(file_location=COPY_FILE_LOCATION, file_format='CSV', table_name='test', task_id=TASK_ID, pattern='folder1/file_[a-g].csv', expression_list=expression, format_options={'header': 'true'}, force_copy=True)
    assert op._create_sql_query() == f"COPY INTO test\nFROM (SELECT {expression} FROM '{COPY_FILE_LOCATION}')\nFILEFORMAT = CSV\nPATTERN = 'folder1/file_[a-g].csv'\nFORMAT_OPTIONS ('header' = 'true')\nCOPY_OPTIONS ('force' = 'true')\n".strip()

def test_copy_with_credential():
    if False:
        print('Hello World!')
    expression = 'col1, col2'
    op = DatabricksCopyIntoOperator(file_location=COPY_FILE_LOCATION, file_format='CSV', table_name='test', task_id=TASK_ID, expression_list=expression, credential={'AZURE_SAS_TOKEN': 'abc'})
    assert op._create_sql_query() == f"COPY INTO test\nFROM (SELECT {expression} FROM '{COPY_FILE_LOCATION}' WITH (CREDENTIAL (AZURE_SAS_TOKEN = 'abc') ))\nFILEFORMAT = CSV\n".strip()

def test_copy_with_target_credential():
    if False:
        return 10
    expression = 'col1, col2'
    op = DatabricksCopyIntoOperator(file_location=COPY_FILE_LOCATION, file_format='CSV', table_name='test', task_id=TASK_ID, expression_list=expression, storage_credential='abc', credential={'AZURE_SAS_TOKEN': 'abc'})
    assert op._create_sql_query() == f"COPY INTO test WITH (CREDENTIAL abc)\nFROM (SELECT {expression} FROM '{COPY_FILE_LOCATION}' WITH (CREDENTIAL (AZURE_SAS_TOKEN = 'abc') ))\nFILEFORMAT = CSV\n".strip()

def test_copy_with_encryption():
    if False:
        print('Hello World!')
    op = DatabricksCopyIntoOperator(file_location=COPY_FILE_LOCATION, file_format='CSV', table_name='test', task_id=TASK_ID, encryption={'TYPE': 'AWS_SSE_C', 'MASTER_KEY': 'abc'})
    assert op._create_sql_query() == f"COPY INTO test\nFROM '{COPY_FILE_LOCATION}' WITH ( ENCRYPTION (TYPE = 'AWS_SSE_C', MASTER_KEY = 'abc'))\nFILEFORMAT = CSV\n".strip()

def test_copy_with_encryption_and_credential():
    if False:
        while True:
            i = 10
    op = DatabricksCopyIntoOperator(file_location=COPY_FILE_LOCATION, file_format='CSV', table_name='test', task_id=TASK_ID, encryption={'TYPE': 'AWS_SSE_C', 'MASTER_KEY': 'abc'}, credential={'AZURE_SAS_TOKEN': 'abc'})
    assert op._create_sql_query() == f"COPY INTO test\nFROM '{COPY_FILE_LOCATION}' WITH (CREDENTIAL (AZURE_SAS_TOKEN = 'abc') ENCRYPTION (TYPE = 'AWS_SSE_C', MASTER_KEY = 'abc'))\nFILEFORMAT = CSV\n".strip()

def test_copy_with_validate_all():
    if False:
        print('Hello World!')
    op = DatabricksCopyIntoOperator(file_location=COPY_FILE_LOCATION, file_format='JSON', table_name='test', task_id=TASK_ID, validate=True)
    assert op._create_sql_query() == f"COPY INTO test\nFROM '{COPY_FILE_LOCATION}'\nFILEFORMAT = JSON\nVALIDATE ALL\n".strip()

def test_copy_with_validate_N_rows():
    if False:
        print('Hello World!')
    op = DatabricksCopyIntoOperator(file_location=COPY_FILE_LOCATION, file_format='JSON', table_name='test', task_id=TASK_ID, validate=10)
    assert op._create_sql_query() == f"COPY INTO test\nFROM '{COPY_FILE_LOCATION}'\nFILEFORMAT = JSON\nVALIDATE 10 ROWS\n".strip()

def test_incorrect_params_files_patterns():
    if False:
        print('Hello World!')
    exception_message = "Only one of 'pattern' or 'files' should be specified"
    with pytest.raises(AirflowException, match=exception_message):
        DatabricksCopyIntoOperator(task_id=TASK_ID, file_location=COPY_FILE_LOCATION, file_format='JSON', table_name='test', files=['file1', 'file2', 'file3'], pattern='abc')

def test_incorrect_params_emtpy_table():
    if False:
        return 10
    exception_message = "table_name shouldn't be empty"
    with pytest.raises(AirflowException, match=exception_message):
        DatabricksCopyIntoOperator(task_id=TASK_ID, file_location=COPY_FILE_LOCATION, file_format='JSON', table_name='')

def test_incorrect_params_emtpy_location():
    if False:
        return 10
    exception_message = "file_location shouldn't be empty"
    with pytest.raises(AirflowException, match=exception_message):
        DatabricksCopyIntoOperator(task_id=TASK_ID, file_location='', file_format='JSON', table_name='abc')

def test_incorrect_params_wrong_format():
    if False:
        while True:
            i = 10
    file_format = 'JSONL'
    exception_message = f"file_format '{file_format}' isn't supported"
    with pytest.raises(AirflowException, match=exception_message):
        DatabricksCopyIntoOperator(task_id=TASK_ID, file_location=COPY_FILE_LOCATION, file_format=file_format, table_name='abc')