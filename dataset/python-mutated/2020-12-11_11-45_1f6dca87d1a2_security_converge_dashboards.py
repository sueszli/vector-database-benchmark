"""security converge dashboards

Revision ID: 1f6dca87d1a2
Revises: 4b84f97828aa
Create Date: 2020-12-11 11:45:25.051084

"""
revision = '1f6dca87d1a2'
down_revision = '4b84f97828aa'
from alembic import op
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from superset.migrations.shared.security_converge import add_pvms, get_reversed_new_pvms, get_reversed_pvm_map, migrate_roles, Pvm
NEW_PVMS = {'Dashboard': ('can_read', 'can_write')}
PVM_MAP = {Pvm('DashboardModelView', 'can_add'): (Pvm('Dashboard', 'can_write'),), Pvm('DashboardModelView', 'can_delete'): (Pvm('Dashboard', 'can_write'),), Pvm('DashboardModelView', 'can_download_dashboards'): (Pvm('Dashboard', 'can_read'),), Pvm('DashboardModelView', 'can_edit'): (Pvm('Dashboard', 'can_write'),), Pvm('DashboardModelView', 'can_favorite_status'): (Pvm('Dashboard', 'can_read'),), Pvm('DashboardModelView', 'can_list'): (Pvm('Dashboard', 'can_read'),), Pvm('DashboardModelView', 'can_mulexport'): (Pvm('Dashboard', 'can_read'),), Pvm('DashboardModelView', 'can_show'): (Pvm('Dashboard', 'can_read'),), Pvm('DashboardModelView', 'muldelete'): (Pvm('Dashboard', 'can_write'),), Pvm('DashboardModelView', 'mulexport'): (Pvm('Dashboard', 'can_read'),), Pvm('DashboardModelViewAsync', 'can_list'): (Pvm('Dashboard', 'can_read'),), Pvm('DashboardModelViewAsync', 'muldelete'): (Pvm('Dashboard', 'can_write'),), Pvm('DashboardModelViewAsync', 'mulexport'): (Pvm('Dashboard', 'can_read'),), Pvm('Dashboard', 'can_new'): (Pvm('Dashboard', 'can_write'),)}

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    bind = op.get_bind()
    session = Session(bind=bind)
    add_pvms(session, NEW_PVMS)
    migrate_roles(session, PVM_MAP)
    try:
        session.commit()
    except SQLAlchemyError as ex:
        print(f'An error occurred while upgrading permissions: {ex}')
        session.rollback()

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    bind = op.get_bind()
    session = Session(bind=bind)
    add_pvms(session, get_reversed_new_pvms(PVM_MAP))
    migrate_roles(session, get_reversed_pvm_map(PVM_MAP))
    try:
        session.commit()
    except SQLAlchemyError as ex:
        print(f'An error occurred while downgrading permissions: {ex}')
        session.rollback()
    pass