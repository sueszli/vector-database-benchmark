import csv
import requests
from dagster import asset
import csv
import requests
from dagster import asset

@asset
def cereals():
    if False:
        while True:
            i = 10
    response = requests.get('https://docs.dagster.io/assets/cereal.csv')
    lines = response.text.split('\n')
    cereal_rows = [row for row in csv.DictReader(lines)]
    return cereal_rows
import csv
import requests
from dagster import asset

@asset
def cereals():
    if False:
        while True:
            i = 10
    response = requests.get('https://docs.dagster.io/assets/cereal.csv')
    lines = response.text.split('\n')
    return [row for row in csv.DictReader(lines)]

@asset
def nabisco_cereals(cereals):
    if False:
        i = 10
        return i + 15
    'Cereals manufactured by Nabisco.'
    return [row for row in cereals if row['mfr'] == 'N']
from dagster import materialize
if __name__ == '__main__':
    materialize([cereals])
from dagster import materialize
if __name__ == '__main__':
    materialize([cereals, nabisco_cereals])