"""Delete table_name unique constraint in mysql

Revision ID: 18532d70ab98
Revises: 3fbbc6e8d654
Create Date: 2020-09-25 10:56:13.711182

"""
revision = '18532d70ab98'
down_revision = '3fbbc6e8d654'
from alembic import op
from sqlalchemy.dialects.mysql.base import MySQLDialect
from sqlalchemy.engine.reflection import Inspector
from superset.utils.core import generic_find_uq_constraint_name

def upgrade():
    if False:
        return 10
    bind = op.get_bind()
    if isinstance(bind.dialect, MySQLDialect):
        constraint_name = generic_find_uq_constraint_name('tables', {'table_name'}, Inspector.from_engine(bind))
        if constraint_name:
            op.drop_constraint(constraint_name, 'tables', type_='unique')

def downgrade():
    if False:
        return 10
    pass