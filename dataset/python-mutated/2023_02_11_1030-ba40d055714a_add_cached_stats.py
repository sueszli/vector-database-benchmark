"""add cached_stats

Revision ID: ba40d055714a
Revises: caee1e8ee0bc
Create Date: 2023-02-11 10:30:21.996198

"""
import sqlalchemy as sa
import sqlmodel
from alembic import op
from sqlalchemy.dialects import postgresql
revision = 'ba40d055714a'
down_revision = 'caee1e8ee0bc'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    op.create_table('cached_stats', sa.Column('name', sqlmodel.sql.sqltypes.AutoString(length=128), nullable=False), sa.Column('modified_date', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False), sa.Column('stats', postgresql.JSONB(astext_type=sa.Text()), nullable=False), sa.PrimaryKeyConstraint('name'))

def downgrade() -> None:
    if False:
        while True:
            i = 10
    op.drop_table('cached_stats')