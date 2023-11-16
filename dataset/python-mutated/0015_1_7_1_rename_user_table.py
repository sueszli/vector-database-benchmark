"""Rename user table

Revision ID: 2e82aab8ef20
Revises: 1968acfc09e3
Create Date: 2016-04-02 19:28:15.211915

"""
from __future__ import annotations
from alembic import op
revision = '2e82aab8ef20'
down_revision = '1968acfc09e3'
branch_labels = None
depends_on = None
airflow_version = '1.7.1'

def upgrade():
    if False:
        print('Hello World!')
    op.rename_table('user', 'users')

def downgrade():
    if False:
        while True:
            i = 10
    op.rename_table('users', 'user')