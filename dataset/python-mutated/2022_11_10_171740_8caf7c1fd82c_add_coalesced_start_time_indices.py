"""Add indices for coalesced `start time` sorts

Revision ID: 8caf7c1fd82c
Revises: 54c1876c68ae
Create Date: 2022-11-10 17:17:40.018108

"""
from alembic import op
revision = '8caf7c1fd82c'
down_revision = '54c1876c68ae'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    with op.get_context().autocommit_block():
        op.execute('\n            CREATE INDEX CONCURRENTLY\n            ix_flow_run__coalesce_start_time_expected_start_time_asc\n            ON flow_run (coalesce(start_time, expected_start_time) ASC);\n            ')
    with op.get_context().autocommit_block():
        op.execute('\n            CREATE INDEX CONCURRENTLY\n            ix_flow_run__coalesce_start_time_expected_start_time_desc\n            ON flow_run (coalesce(start_time, expected_start_time) DESC);\n            ')

def downgrade():
    if False:
        i = 10
        return i + 15
    with op.get_context().autocommit_block():
        op.execute('\n            DROP INDEX CONCURRENTLY ix_flow_run__coalesce_start_time_expected_start_time_desc;\n            ')
        op.execute('\n            DROP INDEX CONCURRENTLY ix_flow_run__coalesce_start_time_expected_start_time_asc;\n            ')