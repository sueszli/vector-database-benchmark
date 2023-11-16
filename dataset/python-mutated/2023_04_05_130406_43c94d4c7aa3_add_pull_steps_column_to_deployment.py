"""Add pull steps column to deployment

Revision ID: 43c94d4c7aa3
Revises: 6a1eb3d442e4
Create Date: 2023-04-05 13:04:06.669728

"""
import sqlalchemy as sa
from alembic import op
import prefect
revision = '43c94d4c7aa3'
down_revision = '6a1eb3d442e4'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('deployment', sa.Column('pull_steps', prefect.server.utilities.database.JSON(astext_type=sa.Text()), nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('deployment', 'pull_steps')