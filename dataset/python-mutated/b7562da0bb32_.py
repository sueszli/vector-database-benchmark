"""project.path unique constraint

Revision ID: b7562da0bb32
Revises: 40a642486f67
Create Date: 2020-12-22 16:42:34.758801

"""
import sqlalchemy as sa
from alembic import op
revision = 'b7562da0bb32'
down_revision = '40a642486f67'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    op.create_unique_constraint(op.f('uq_project_path'), 'project', ['path'])

def downgrade():
    if False:
        return 10
    op.drop_constraint(op.f('uq_project_path'), 'project', type_='unique')