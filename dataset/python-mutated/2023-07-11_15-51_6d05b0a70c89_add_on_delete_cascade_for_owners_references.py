"""add on delete cascade for owners references

Revision ID: 6d05b0a70c89
Revises: f92a3124dd66
Create Date: 2023-07-11 15:51:57.964635

"""
revision = '6d05b0a70c89'
down_revision = 'f92a3124dd66'
from superset.migrations.shared.constraints import ForeignKey, redefine
foreign_keys = [ForeignKey(table='dashboard_user', referent_table='ab_user', local_cols=['user_id'], remote_cols=['id']), ForeignKey(table='dashboard_user', referent_table='dashboards', local_cols=['dashboard_id'], remote_cols=['id']), ForeignKey(table='report_schedule_user', referent_table='ab_user', local_cols=['user_id'], remote_cols=['id']), ForeignKey(table='report_schedule_user', referent_table='report_schedule', local_cols=['report_schedule_id'], remote_cols=['id']), ForeignKey(table='slice_user', referent_table='ab_user', local_cols=['user_id'], remote_cols=['id']), ForeignKey(table='slice_user', referent_table='slices', local_cols=['slice_id'], remote_cols=['id'])]

def upgrade():
    if False:
        print('Hello World!')
    for foreign_key in foreign_keys:
        redefine(foreign_key, on_delete='CASCADE')

def downgrade():
    if False:
        return 10
    for foreign_key in foreign_keys:
        redefine(foreign_key)