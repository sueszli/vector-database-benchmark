"""security converge reports

Revision ID: 40f16acf1ba7
Revises: e38177dbf641
Create Date: 2020-11-30 15:25:47.489419

"""
revision = '40f16acf1ba7'
down_revision = '5daced1f0e76'
from alembic import op
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from superset.migrations.shared.security_converge import add_pvms, get_reversed_new_pvms, get_reversed_pvm_map, migrate_roles, Pvm
NEW_PVMS = {'ReportSchedule': ('can_read', 'can_write')}
PVM_MAP = {Pvm('ReportSchedule', 'can_list'): (Pvm('ReportSchedule', 'can_read'),), Pvm('ReportSchedule', 'can_show'): (Pvm('ReportSchedule', 'can_read'),), Pvm('ReportSchedule', 'can_add'): (Pvm('ReportSchedule', 'can_write'),), Pvm('ReportSchedule', 'can_edit'): (Pvm('ReportSchedule', 'can_write'),), Pvm('ReportSchedule', 'can_delete'): (Pvm('ReportSchedule', 'can_write'),)}

def upgrade():
    if False:
        i = 10
        return i + 15
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
        i = 10
        return i + 15
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