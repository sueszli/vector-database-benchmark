import requests
from dagster import DagsterType, In, Out, get_dagster_logger, job, op

def is_list_of_dicts(_, value):
    if False:
        return 10
    return isinstance(value, list) and all((isinstance(element, dict) for element in value))
SimpleDataFrame = DagsterType(name='SimpleDataFrame', type_check_fn=is_list_of_dicts, description='A naive representation of a data frame, e.g., as returned by csv.DictReader.')

@op(out=Out(SimpleDataFrame))
def bad_download_csv():
    if False:
        return 10
    response = requests.get('https://docs.dagster.io/assets/cereal.csv')
    lines = response.text.split('\n')
    get_dagster_logger().info(f'Read {len(lines)} lines')
    return ['not_a_dict']

@op(ins={'cereals': In(SimpleDataFrame)})
def sort_by_calories(cereals):
    if False:
        return 10
    sorted_cereals = sorted(cereals, key=lambda cereal: cereal['calories'])
    get_dagster_logger().info(f"Most caloric cereal: {sorted_cereals[-1]['name']}")

@job
def custom_type_job():
    if False:
        for i in range(10):
            print('nop')
    sort_by_calories(bad_download_csv())