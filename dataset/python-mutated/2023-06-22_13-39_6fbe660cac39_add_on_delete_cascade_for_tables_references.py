"""add on delete cascade for tables references

Revision ID: 6fbe660cac39
Revises: 90139bf715e4
Create Date: 2023-06-22 13:39:47.989373

"""
revision = '6fbe660cac39'
down_revision = '90139bf715e4'
from superset.migrations.shared.constraints import ForeignKey, redefine
foreign_keys = [ForeignKey(table='sql_metrics', referent_table='tables', local_cols=['table_id'], remote_cols=['id']), ForeignKey(table='table_columns', referent_table='tables', local_cols=['table_id'], remote_cols=['id']), ForeignKey(table='sqlatable_user', referent_table='ab_user', local_cols=['user_id'], remote_cols=['id']), ForeignKey(table='sqlatable_user', referent_table='tables', local_cols=['table_id'], remote_cols=['id'])]

def upgrade():
    if False:
        while True:
            i = 10
    for foreign_key in foreign_keys:
        redefine(foreign_key, on_delete='CASCADE')

def downgrade():
    if False:
        print('Hello World!')
    for foreign_key in foreign_keys:
        redefine(foreign_key)