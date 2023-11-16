"""Add InteractivePipelineRun.pipeline_definition field

Revision ID: 23def7128481
Revises: 637920f5715f
Create Date: 2022-05-16 11:40:02.148006

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = '23def7128481'
down_revision = '637920f5715f'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    op.add_column('pipeline_runs', sa.Column('pipeline_definition', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False))

def downgrade():
    if False:
        return 10
    op.drop_column('pipeline_runs', 'pipeline_definition')