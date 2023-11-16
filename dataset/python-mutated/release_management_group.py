from __future__ import annotations
import click
from airflow_breeze.utils.click_utils import BreezeGroup

@click.group(cls=BreezeGroup, name='release-management', help='Tools that release managers can use to prepare and manage Airflow releases')
def release_management():
    if False:
        for i in range(10):
            print('nop')
    pass