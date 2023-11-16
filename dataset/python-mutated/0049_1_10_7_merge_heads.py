"""Straighten out the migrations

Revision ID: 08364691d074
Revises: a56c9515abdc, 004c1210f153, 74effc47d867, b3b105409875
Create Date: 2019-11-19 22:05:11.752222

"""
from __future__ import annotations
revision = '08364691d074'
down_revision = ('a56c9515abdc', '004c1210f153', '74effc47d867', 'b3b105409875')
branch_labels = None
depends_on = None
airflow_version = '1.10.7'

def upgrade():
    if False:
        print('Hello World!')
    pass

def downgrade():
    if False:
        while True:
            i = 10
    pass