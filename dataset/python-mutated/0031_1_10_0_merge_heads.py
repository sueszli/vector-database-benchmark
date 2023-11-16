"""Merge migrations Heads

Revision ID: 05f30312d566
Revises: 86770d1215c0, 0e2a74e0fc9f
Create Date: 2018-06-17 10:47:23.339972

"""
from __future__ import annotations
revision = '05f30312d566'
down_revision = ('86770d1215c0', '0e2a74e0fc9f')
branch_labels = None
depends_on = None
airflow_version = '1.10.0'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    pass

def downgrade():
    if False:
        print('Hello World!')
    pass