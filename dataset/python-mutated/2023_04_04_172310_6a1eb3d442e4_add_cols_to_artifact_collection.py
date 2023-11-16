"""Add cols to artifact_collection

Revision ID: 6a1eb3d442e4
Revises: 3bf47e3ce2dd
Create Date: 2023-03-31 17:23:10.947068

"""
import sqlalchemy as sa
from alembic import op
import prefect
revision = '6a1eb3d442e4'
down_revision = '3bf47e3ce2dd'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column(table_name='artifact_collection', column=sa.Column('task_run_id', prefect.server.utilities.database.UUID(), nullable=True))
    op.add_column(table_name='artifact_collection', column=sa.Column('flow_run_id', prefect.server.utilities.database.UUID(), nullable=True))
    op.add_column(table_name='artifact_collection', column=sa.Column('type', sa.String(), nullable=True))
    op.add_column(table_name='artifact_collection', column=sa.Column('data', sa.JSON(), nullable=True))
    op.add_column(table_name='artifact_collection', column=sa.Column('description', sa.String(), nullable=True))
    op.add_column(table_name='artifact_collection', column=sa.Column('metadata_', sa.JSON(), nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column(table_name='artifact_collection', column_name='metadata_')
    op.drop_column(table_name='artifact_collection', column_name='description')
    op.drop_column(table_name='artifact_collection', column_name='data')
    op.drop_column(table_name='artifact_collection', column_name='type')
    op.drop_column(table_name='artifact_collection', column_name='flow_run_id')
    op.drop_column(table_name='artifact_collection', column_name='task_run_id')