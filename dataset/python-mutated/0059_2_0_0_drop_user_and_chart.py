"""Drop ``user`` and ``chart`` table

Revision ID: cf5dc11e79ad
Revises: 03afc6b6f902
Create Date: 2019-01-24 15:30:35.834740

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect, text
from sqlalchemy.dialects import mysql
revision = 'cf5dc11e79ad'
down_revision = '03afc6b6f902'
branch_labels = None
depends_on = None
airflow_version = '2.0.0'

def upgrade():
    if False:
        return 10
    conn = op.get_bind()
    inspector = inspect(conn)
    tables = inspector.get_table_names()
    if 'known_event' in tables:
        for fkey in inspector.get_foreign_keys(table_name='known_event', referred_table='users'):
            if fkey['name']:
                with op.batch_alter_table(table_name='known_event') as bop:
                    bop.drop_constraint(fkey['name'], type_='foreignkey')
    if 'chart' in tables:
        op.drop_table('chart')
    if 'users' in tables:
        op.drop_table('users')

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    conn = op.get_bind()
    op.create_table('users', sa.Column('id', sa.Integer(), nullable=False), sa.Column('username', sa.String(length=250), nullable=True), sa.Column('email', sa.String(length=500), nullable=True), sa.Column('password', sa.String(255)), sa.Column('superuser', sa.Boolean(), default=False), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('username'))
    op.create_table('chart', sa.Column('id', sa.Integer(), nullable=False), sa.Column('label', sa.String(length=200), nullable=True), sa.Column('conn_id', sa.String(length=250), nullable=False), sa.Column('user_id', sa.Integer(), nullable=True), sa.Column('chart_type', sa.String(length=100), nullable=True), sa.Column('sql_layout', sa.String(length=50), nullable=True), sa.Column('sql', sa.Text(), nullable=True), sa.Column('y_log_scale', sa.Boolean(), nullable=True), sa.Column('show_datatable', sa.Boolean(), nullable=True), sa.Column('show_sql', sa.Boolean(), nullable=True), sa.Column('height', sa.Integer(), nullable=True), sa.Column('default_params', sa.String(length=5000), nullable=True), sa.Column('x_is_date', sa.Boolean(), nullable=True), sa.Column('iteration_no', sa.Integer(), nullable=True), sa.Column('last_modified', sa.DateTime(), nullable=True), sa.ForeignKeyConstraint(['user_id'], ['users.id']), sa.PrimaryKeyConstraint('id'))
    if conn.dialect.name == 'mysql':
        conn.execute(text("SET time_zone = '+00:00'"))
        op.alter_column(table_name='chart', column_name='last_modified', type_=mysql.TIMESTAMP(fsp=6))
    else:
        if conn.dialect.name in ('sqlite', 'mssql'):
            return
        if conn.dialect.name == 'postgresql':
            conn.execute(text('set timezone=UTC'))
        op.alter_column(table_name='chart', column_name='last_modified', type_=sa.TIMESTAMP(timezone=True))