"""Fix description field in ``connection`` to be ``text``

Revision ID: 64a7d6477aae
Revises: f5b5ec089444
Create Date: 2020-11-25 08:56:11.866607

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = '64a7d6477aae'
down_revision = '61ec73d9401f'
branch_labels = None
depends_on = None
airflow_version = '2.0.0'

def upgrade():
    if False:
        return 10
    'Apply Fix description field in ``connection`` to be ``text``'
    conn = op.get_bind()
    if conn.dialect.name == 'sqlite':
        return
    if conn.dialect.name == 'mysql':
        op.alter_column('connection', 'description', existing_type=sa.String(length=5000), type_=sa.Text(length=5000), existing_nullable=True)
    else:
        op.alter_column('connection', 'description', existing_type=sa.String(length=5000), type_=sa.Text())

def downgrade():
    if False:
        print('Hello World!')
    'Unapply Fix description field in ``connection`` to be ``text``'
    conn = op.get_bind()
    if conn.dialect.name == 'sqlite':
        return
    if conn.dialect.name == 'mysql':
        op.alter_column('connection', 'description', existing_type=sa.Text(5000), type_=sa.String(length=5000), existing_nullable=True)
    else:
        op.alter_column('connection', 'description', existing_type=sa.Text(), type_=sa.String(length=5000), existing_nullable=True)