import csv
import requests
from dagster import OpExecutionContext, job, op

@op
def download_cereals():
    if False:
        return 10
    response = requests.get('https://docs.dagster.io/assets/cereal.csv')
    lines = response.text.split('\n')
    return [row for row in csv.DictReader(lines)]

@op
def find_sugariest(context: OpExecutionContext, cereals):
    if False:
        return 10
    sorted_by_sugar = sorted(cereals, key=lambda cereal: cereal['sugars'])
    context.log.info(f"{sorted_by_sugar[-1]['name']} is the sugariest cereal")

@job
def hello_cereal_job():
    if False:
        i = 10
        return i + 15
    find_sugariest(download_cereals())