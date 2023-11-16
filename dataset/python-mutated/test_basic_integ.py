import pytest
from dagster import In, Output, job, op
from dagster._utils import file_relative_path
from dagster_ge.factory import ge_data_context, ge_validation_op_factory, ge_validation_op_factory_v3
from dagster_pyspark import DataFrame as DagsterPySparkDataFrame, pyspark_resource
from pandas import read_csv

@op
def pandas_yielder(_):
    if False:
        while True:
            i = 10
    return read_csv(file_relative_path(__file__, './basic.csv'))

@op(required_resource_keys={'pyspark'})
def pyspark_yielder(context):
    if False:
        print('Hello World!')
    return context.resources.pyspark.spark_session.read.format('csv').options(header='true', inferSchema='true').load(file_relative_path(__file__, './basic.csv'))

@op(ins={'res': In()})
def reyielder(_context, res):
    if False:
        for i in range(10):
            print('nop')
    yield Output((res['statistics'], res['results']))

@job(resource_defs={'ge_data_context': ge_data_context})
def hello_world_pandas_job_v2():
    if False:
        i = 10
        return i + 15
    reyielder(ge_validation_op_factory('ge_validation_op', 'getest', 'basic.warning')(pandas_yielder()))

@job(resource_defs={'ge_data_context': ge_data_context})
def hello_world_pandas_job_v3():
    if False:
        return 10
    reyielder(ge_validation_op_factory_v3(name='ge_validation_op', datasource_name='getest', data_connector_name='my_runtime_data_connector', data_asset_name='test_asset', suite_name='basic.warning', batch_identifiers={'foo': 'bar'})(pandas_yielder()))

@job(resource_defs={'ge_data_context': ge_data_context, 'pyspark': pyspark_resource})
def hello_world_pyspark_job():
    if False:
        for i in range(10):
            print('nop')
    validate = ge_validation_op_factory('ge_validation_op', 'getestspark', 'basic.warning', input_dagster_type=DagsterPySparkDataFrame)
    reyielder(validate(pyspark_yielder()))

@pytest.mark.parametrize('job_def, ge_dir', [(hello_world_pandas_job_v2, './great_expectations'), (hello_world_pandas_job_v3, './great_expectations_v3')])
def test_yielded_results_config_pandas(snapshot, job_def, ge_dir):
    if False:
        print('Hello World!')
    run_config = {'resources': {'ge_data_context': {'config': {'ge_root_dir': file_relative_path(__file__, ge_dir)}}}}
    result = job_def.execute_in_process(run_config=run_config)
    assert result.output_for_node('reyielder')[0]['success_percent'] == 100
    expectations = result.expectation_results_for_node('ge_validation_op')
    assert len(expectations) == 1
    mainexpect = expectations[0]
    assert mainexpect.success
    metadata = mainexpect.metadata['Expectation Results'].md_str.split('### Info')[0]
    snapshot.assert_match(metadata)

def test_yielded_results_config_pyspark_v2(snapshot):
    if False:
        return 10
    run_config = {'resources': {'ge_data_context': {'config': {'ge_root_dir': file_relative_path(__file__, './great_expectations')}}}}
    result = hello_world_pyspark_job.execute_in_process(run_config=run_config)
    assert result.output_for_node('reyielder')[0]['success_percent'] == 100
    expectations = result.expectation_results_for_node('ge_validation_op')
    assert len(expectations) == 1
    mainexpect = expectations[0]
    assert mainexpect.success
    metadata = mainexpect.metadata['Expectation Results'].md_str.split('### Info')[0]
    snapshot.assert_match(metadata)