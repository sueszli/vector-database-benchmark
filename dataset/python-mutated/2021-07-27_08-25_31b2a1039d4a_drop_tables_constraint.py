"""drop tables constraint

Revision ID: 31b2a1039d4a
Revises: ae1ed299413b
Create Date: 2021-07-27 08:25:20.755453

"""
from alembic import op
from sqlalchemy import engine
from sqlalchemy.exc import OperationalError, ProgrammingError
from superset.utils.core import generic_find_uq_constraint_name
revision = '31b2a1039d4a'
down_revision = 'ae1ed299413b'
conv = {'uq': 'uq_%(table_name)s_%(column_0_name)s'}

def upgrade():
    if False:
        i = 10
        return i + 15
    bind = op.get_bind()
    insp = engine.reflection.Inspector.from_engine(bind)
    if (constraint := generic_find_uq_constraint_name('tables', {'table_name'}, insp)):
        with op.batch_alter_table('tables', naming_convention=conv) as batch_op:
            batch_op.drop_constraint(constraint, type_='unique')

def downgrade():
    if False:
        i = 10
        return i + 15
    pass