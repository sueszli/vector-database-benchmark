import csv
import requests
from dagster import get_dagster_logger, job, op

@op
def download_csv():
    if False:
        print('Hello World!')
    response = requests.get('https://docs.dagster.io/assets/cereal.csv')
    lines = response.text.split('\n')
    get_dagster_logger().info(f'Read {len(lines)} lines')
    return [row for row in csv.DictReader(lines)]

@op
def sort_by_calories(cereals):
    if False:
        i = 10
        return i + 15
    sorted_cereals = sorted(cereals, key=lambda cereal: cereal['calories'])
    logger = get_dagster_logger()
    logger.info('Least caloric cereal: {least_caloric}'.format(least_caloric=sorted_cereals[0]['name']))
    logger.info('Most caloric cereal: {most_caloric}'.format(most_caloric=sorted_cereals[-1]['name']))

@job
def inputs_job():
    if False:
        return 10
    sort_by_calories(download_csv())
if __name__ == '__main__':
    result = inputs_job.execute_in_process()
    assert result.success