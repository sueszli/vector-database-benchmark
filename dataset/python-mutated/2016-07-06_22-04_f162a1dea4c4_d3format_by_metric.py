"""d3format_by_metric

Revision ID: f162a1dea4c4
Revises: 960c69cb1f5b
Create Date: 2016-07-06 22:04:28.685100

"""
revision = 'f162a1dea4c4'
down_revision = '960c69cb1f5b'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('metrics', sa.Column('d3format', sa.String(length=128), nullable=True))
    op.add_column('sql_metrics', sa.Column('d3format', sa.String(length=128), nullable=True))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('sql_metrics', 'd3format')
    op.drop_column('metrics', 'd3format')