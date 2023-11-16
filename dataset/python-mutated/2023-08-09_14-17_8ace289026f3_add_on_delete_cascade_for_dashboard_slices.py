"""add on delete cascade for dashboard_slices

Revision ID: 8ace289026f3
Revises: 2e826adca42c
Create Date: 2023-08-09 14:17:53.326191

"""
revision = '8ace289026f3'
down_revision = '2e826adca42c'
from superset.migrations.shared.constraints import ForeignKey, redefine
foreign_keys = [ForeignKey(table='dashboard_slices', referent_table='dashboards', local_cols=['dashboard_id'], remote_cols=['id']), ForeignKey(table='dashboard_slices', referent_table='slices', local_cols=['slice_id'], remote_cols=['id'])]

def upgrade():
    if False:
        return 10
    for foreign_key in foreign_keys:
        redefine(foreign_key, on_delete='CASCADE')

def downgrade():
    if False:
        while True:
            i = 10
    for foreign_key in foreign_keys:
        redefine(foreign_key)