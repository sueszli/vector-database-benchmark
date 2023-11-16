"""security converge databases

Revision ID: 42b4c9e01447
Revises: 5daced1f0e76
Create Date: 2020-12-14 10:49:36.110805

"""
revision = '42b4c9e01447'
down_revision = '1f6dca87d1a2'
import sqlalchemy as sa
from alembic import op
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from superset.migrations.shared.security_converge import add_pvms, get_reversed_new_pvms, get_reversed_pvm_map, migrate_roles, Pvm
NEW_PVMS = {'Database': ('can_read', 'can_write')}
PVM_MAP = {Pvm('DatabaseView', 'can_add'): (Pvm('Database', 'can_write'),), Pvm('DatabaseView', 'can_delete'): (Pvm('Database', 'can_write'),), Pvm('DatabaseView', 'can_edit'): (Pvm('Database', 'can_write'),), Pvm('DatabaseView', 'can_list'): (Pvm('Database', 'can_read'),), Pvm('DatabaseView', 'can_mulexport'): (Pvm('Database', 'can_read'),), Pvm('DatabaseView', 'can_post'): (Pvm('Database', 'can_write'),), Pvm('DatabaseView', 'can_show'): (Pvm('Database', 'can_read'),), Pvm('DatabaseView', 'muldelete'): (Pvm('Database', 'can_write'),), Pvm('DatabaseView', 'yaml_export'): (Pvm('Database', 'can_read'),)}

def upgrade():
    if False:
        print('Hello World!')
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
        while True:
            i = 10
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