from dagster import AssetMaterialization, ExpectationResult, Failure, MetadataValue, Out, Output, RetryRequested, op

def do_some_transform(_):
    if False:
        return 10
    return []

def calculate_bytes(_):
    if False:
        for i in range(10):
            print('nop')
    return 0.0

def get_some_data():
    if False:
        for i in range(10):
            print('nop')
    return []

def some_calculation(_):
    if False:
        return 10
    return 0

def get_files(_path):
    if False:
        for i in range(10):
            print('nop')
    return []

def store_to_s3(_):
    if False:
        for i in range(10):
            print('nop')
    return

def flaky_operation():
    if False:
        while True:
            i = 10
    return 0
from dagster import MetadataValue, Output, op, OpExecutionContext

@op
def my_metadata_output(context: OpExecutionContext) -> Output:
    if False:
        for i in range(10):
            print('nop')
    df = get_some_data()
    return Output(df, metadata={'text_metadata': 'Text-based metadata for this event', 'dashboard_url': MetadataValue.url('http://mycoolsite.com/url_for_my_data'), 'raw_count': len(df), 'size (bytes)': calculate_bytes(df)})
from dagster import Output, op
from typing import Tuple

@op
def my_output_op() -> Output:
    if False:
        print('Hello World!')
    return Output('some_value', metadata={'some_metadata': 'a_value'})

@op
def my_output_generic_op() -> Output[int]:
    if False:
        while True:
            i = 10
    return Output(5, metadata={'some_metadata': 'a_value'})

@op(out={'int_out': Out(), 'str_out': Out()})
def my_multiple_generic_output_op() -> Tuple[Output[int], Output[str]]:
    if False:
        for i in range(10):
            print('nop')
    return (Output(5, metadata={'some_metadata': 'a_value'}), Output('foo', metadata={'some_metadata': 'another_value'}))
from dagster import ExpectationResult, MetadataValue, op, OpExecutionContext

@op
def my_metadata_expectation_op(context: OpExecutionContext, df):
    if False:
        print('Hello World!')
    df = do_some_transform(df)
    context.log_event(ExpectationResult(success=len(df) > 0, description='ensure dataframe has rows', metadata={'text_metadata': 'Text-based metadata for this event', 'dashboard_url': MetadataValue.url('http://mycoolsite.com/url_for_my_data'), 'raw_count': len(df), 'size (bytes)': calculate_bytes(df)}))
    return df
from dagster import Failure, op, MetadataValue

@op
def my_failure_op():
    if False:
        return 10
    path = '/path/to/files'
    my_files = get_files(path)
    if len(my_files) == 0:
        raise Failure(description='No files to process', metadata={'filepath': MetadataValue.path(path), 'dashboard_url': MetadataValue.url('http://mycoolsite.com/failures')})
    return some_calculation(my_files)
from dagster import RetryRequested, op

@op
def my_retry_op():
    if False:
        while True:
            i = 10
    try:
        result = flaky_operation()
    except Exception as e:
        raise RetryRequested(max_retries=3) from e
    return result
from dagster import AssetMaterialization, op, OpExecutionContext

@op
def my_asset_op(context: OpExecutionContext):
    if False:
        return 10
    df = get_some_data()
    store_to_s3(df)
    context.log_event(AssetMaterialization(asset_key='s3.my_asset', description='A df I stored in s3'))
    result = do_some_transform(df)
    return result
from dagster import AssetMaterialization, Output, op

@op
def my_asset_op_yields():
    if False:
        for i in range(10):
            print('nop')
    df = get_some_data()
    store_to_s3(df)
    yield AssetMaterialization(asset_key='s3.my_asset', description='A df I stored in s3')
    result = do_some_transform(df)
    yield Output(result)
from dagster import ExpectationResult, op, OpExecutionContext

@op
def my_expectation_op(context: OpExecutionContext, df):
    if False:
        return 10
    do_some_transform(df)
    context.log_event(ExpectationResult(success=len(df) > 0, description='ensure dataframe has rows'))
    return df
from dagster import Output, op

@op(out={'out1': Out(str), 'out2': Out(int)})
def my_op_yields():
    if False:
        return 10
    yield Output(5, output_name='out2')
    yield Output('foo', output_name='out1')