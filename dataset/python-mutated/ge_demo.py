from dagster import job, op
from dagster._utils import file_relative_path
from dagster_ge.factory import ge_data_context, ge_validation_op_factory
from pandas import read_csv

@op
def read_in_datafile(csv_path):
    if False:
        while True:
            i = 10
    return read_csv(csv_path)

@op
def process_payroll(df):
    if False:
        for i in range(10):
            print('nop')
    return len(df)

@op
def postprocess_payroll(numrows, expectation):
    if False:
        for i in range(10):
            print('nop')
    if expectation['success']:
        return numrows
    else:
        raise ValueError
payroll_expectations = ge_validation_op_factory(name='ge_validation_op', datasource_name='getest', suite_name='basic.warning')

@job(resource_defs={'ge_data_context': ge_data_context}, config={'resources': {'ge_data_context': {'config': {'ge_root_dir': file_relative_path(__file__, './great_expectations')}}}, 'solids': {'read_in_datafile': {'inputs': {'csv_path': {'value': file_relative_path(__file__, './succeed.csv')}}}}})
def payroll_data():
    if False:
        for i in range(10):
            print('nop')
    output_df = read_in_datafile()
    postprocess_payroll(process_payroll(output_df), payroll_expectations(output_df))