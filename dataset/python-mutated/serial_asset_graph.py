import csv
import requests
from dagster import asset

@asset
def cereals():
    if False:
        i = 10
        return i + 15
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