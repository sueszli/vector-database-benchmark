"""security converge logs

Revision ID: 4b84f97828aa
Revises: 45731db65d9c
Create Date: 2020-12-14 13:40:46.492449

"""
from alembic import op
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from superset.migrations.shared.security_converge import add_pvms, get_reversed_new_pvms, get_reversed_pvm_map, migrate_roles, Pvm
revision = '4b84f97828aa'
down_revision = '45731db65d9c'
NEW_PVMS = {'Log': ('can_read', 'can_write')}
PVM_MAP = {Pvm('LogModelView', 'can_show'): (Pvm('Log', 'can_read'),), Pvm('LogModelView', 'can_add'): (Pvm('Log', 'can_write'),), Pvm('LogModelView', 'can_list'): (Pvm('Log', 'can_read'),)}

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
        print(f'An error occurred while upgrading Logs permissions: {ex}')
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
        print(f'An error occurred while downgrading Logs permissions: {ex}')
        session.rollback()
    pass