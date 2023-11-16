"""Maintain history for compatibility with earlier migrations

Revision ID: 13eb55f81627
Revises: 1507a7289a2f
Create Date: 2015-08-23 05:12:49.732174

"""
from __future__ import annotations
revision = '13eb55f81627'
down_revision = '1507a7289a2f'
branch_labels = None
depends_on = None
airflow_version = '1.5.0'

def upgrade():
    if False:
        return 10
    pass

def downgrade():
    if False:
        while True:
            i = 10
    pass