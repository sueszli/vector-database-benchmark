"""drop rouge constraints and tables

Revision ID: f92a3124dd66
Revises: 240d23c7f86f
Create Date: 2023-07-07 20:06:22.659096

"""
revision = 'f92a3124dd66'
down_revision = '240d23c7f86f'
from alembic import op
from sqlalchemy.engine.reflection import Inspector
from superset.utils.core import generic_find_fk_constraint_name

def upgrade():
    if False:
        i = 10
        return i + 15
    bind = op.get_bind()
    insp = Inspector.from_engine(bind)
    tables = insp.get_table_names()
    conv = {'fk': 'fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s'}
    if 'datasources' in tables:
        with op.batch_alter_table('slices', naming_convention=conv) as batch_op:
            if (constraint := generic_find_fk_constraint_name(table='slices', columns={'id'}, referenced='datasources', insp=insp)):
                batch_op.drop_constraint(constraint, type_='foreignkey')
    for table in ['alert_logs', 'alert_owner', 'sql_observations', 'alerts', 'columns', 'metrics', 'druiddatasource_user', 'datasources', 'clusters', 'dashboard_email_schedules', 'slice_email_schedules']:
        if table in tables:
            op.drop_table(table)

def downgrade():
    if False:
        return 10
    pass