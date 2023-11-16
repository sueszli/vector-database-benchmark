"""security converge queries

Revision ID: e37912a26567
Revises: 42b4c9e01447
Create Date: 2020-12-16 12:15:28.291777

"""
revision = 'e37912a26567'
down_revision = '42b4c9e01447'
from alembic import op
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from superset.migrations.shared.security_converge import add_pvms, get_reversed_new_pvms, get_reversed_pvm_map, migrate_roles, Pvm
NEW_PVMS = {'Query': ('can_read',)}
PVM_MAP = {Pvm('QueryView', 'can_list'): (Pvm('Query', 'can_read'),), Pvm('QueryView', 'can_show'): (Pvm('Query', 'can_read'),)}

def upgrade():
    if False:
        while True:
            i = 10
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