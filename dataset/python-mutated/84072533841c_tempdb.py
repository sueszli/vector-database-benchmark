"""TempDB

Revision ID: 84072533841c
Revises: 6f9da26137b5
Create Date: 2023-01-31 23:22:12.564277

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
revision = '84072533841c'
down_revision = '6f9da26137b5'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.drop_index('ix_product_key', table_name='product')
    op.drop_table('product')

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.create_table('product', sa.Column('key', sa.VARCHAR(), autoincrement=False, nullable=False), sa.Column('code', sa.INTEGER(), autoincrement=False, nullable=True), sa.Column('codebar', sa.VARCHAR(), autoincrement=False, nullable=True), sa.Column('codebarinner', sa.VARCHAR(), autoincrement=False, nullable=True), sa.Column('codebarmaster', sa.VARCHAR(), autoincrement=False, nullable=True), sa.Column('unit', sa.VARCHAR(), autoincrement=False, nullable=False), sa.Column('description', sa.VARCHAR(), autoincrement=False, nullable=False), sa.Column('brand', sa.VARCHAR(), autoincrement=False, nullable=True), sa.Column('buy', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False), sa.Column('retailsale', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False), sa.Column('wholesale', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False), sa.Column('inventory', sa.INTEGER(), autoincrement=False, nullable=True), sa.Column('min_inventory', sa.INTEGER(), autoincrement=False, nullable=True), sa.Column('department', sa.VARCHAR(), autoincrement=False, nullable=True), sa.Column('id', sa.INTEGER(), autoincrement=False, nullable=True), sa.Column('LastUpdate', postgresql.TIMESTAMP(), autoincrement=False, nullable=True), sa.PrimaryKeyConstraint('key', name='product_pkey'), sa.UniqueConstraint('code', name='product_code_key'), sa.UniqueConstraint('codebar', name='product_codebar_key'), sa.UniqueConstraint('codebarinner', name='product_codebarinner_key'), sa.UniqueConstraint('codebarmaster', name='product_codebarmaster_key'))
    op.create_index('ix_product_key', 'product', ['key'], unique=False)