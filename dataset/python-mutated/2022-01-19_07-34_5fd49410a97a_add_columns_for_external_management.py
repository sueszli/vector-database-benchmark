"""Add columns for external management

Revision ID: 5fd49410a97a
Revises: c53bae8f08dd
Create Date: 2022-01-19 07:34:20.594786

"""
revision = '5fd49410a97a'
down_revision = 'c53bae8f08dd'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        return 10
    with op.batch_alter_table('dashboards') as batch_op:
        batch_op.add_column(sa.Column('is_managed_externally', sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column('external_url', sa.Text(), nullable=True))
    with op.batch_alter_table('datasources') as batch_op:
        batch_op.add_column(sa.Column('is_managed_externally', sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column('external_url', sa.Text(), nullable=True))
    with op.batch_alter_table('dbs') as batch_op:
        batch_op.add_column(sa.Column('is_managed_externally', sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column('external_url', sa.Text(), nullable=True))
    with op.batch_alter_table('slices') as batch_op:
        batch_op.add_column(sa.Column('is_managed_externally', sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column('external_url', sa.Text(), nullable=True))
    with op.batch_alter_table('tables') as batch_op:
        batch_op.add_column(sa.Column('is_managed_externally', sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column('external_url', sa.Text(), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('tables') as batch_op:
        batch_op.drop_column('external_url')
        batch_op.drop_column('is_managed_externally')
    with op.batch_alter_table('slices') as batch_op:
        batch_op.drop_column('external_url')
        batch_op.drop_column('is_managed_externally')
    with op.batch_alter_table('dbs') as batch_op:
        batch_op.drop_column('external_url')
        batch_op.drop_column('is_managed_externally')
    with op.batch_alter_table('datasources') as batch_op:
        batch_op.drop_column('external_url')
        batch_op.drop_column('is_managed_externally')
    with op.batch_alter_table('dashboards') as batch_op:
        batch_op.drop_column('external_url')
        batch_op.drop_column('is_managed_externally')