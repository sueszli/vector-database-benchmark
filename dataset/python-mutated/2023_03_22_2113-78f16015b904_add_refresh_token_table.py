"""Add refresh token table

Revision ID: 78f16015b904
Revises: 629d5081160f
Create Date: 2023-03-22 21:13:41.411718

"""
import sqlalchemy as sa
import sqlmodel
from alembic import op
revision = '78f16015b904'
down_revision = '629d5081160f'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.create_table('refresh_token', sa.Column('token_hash', sqlmodel.sql.sqltypes.AutoString(), nullable=False), sa.Column('user_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False), sa.Column('enabled', sa.Boolean(), nullable=False), sa.ForeignKeyConstraint(['user_id'], ['user.id']), sa.PrimaryKeyConstraint('token_hash'))
    op.create_index(op.f('ix_refresh_token_user_id'), 'refresh_token', ['user_id'], unique=False)

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.drop_index(op.f('ix_refresh_token_user_id'), table_name='refresh_token')
    op.drop_table('refresh_token')