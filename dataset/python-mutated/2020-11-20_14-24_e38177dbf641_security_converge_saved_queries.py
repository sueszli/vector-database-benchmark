"""security converge saved queries

Revision ID: e38177dbf641
Revises: a8173232b786
Create Date: 2020-11-20 14:24:03.643031

"""
revision = 'e38177dbf641'
down_revision = 'a8173232b786'
from alembic import op
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from superset.migrations.shared.security_converge import add_pvms, get_reversed_new_pvms, get_reversed_pvm_map, migrate_roles, Pvm
NEW_PVMS = {'SavedQuery': ('can_read', 'can_write')}
PVM_MAP = {Pvm('SavedQueryView', 'can_list'): (Pvm('SavedQuery', 'can_read'),), Pvm('SavedQueryView', 'can_show'): (Pvm('SavedQuery', 'can_read'),), Pvm('SavedQueryView', 'can_add'): (Pvm('SavedQuery', 'can_write'),), Pvm('SavedQueryView', 'can_edit'): (Pvm('SavedQuery', 'can_write'),), Pvm('SavedQueryView', 'can_delete'): (Pvm('SavedQuery', 'can_write'),), Pvm('SavedQueryView', 'muldelete'): (Pvm('SavedQuery', 'can_write'),), Pvm('SavedQueryView', 'can_mulexport'): (Pvm('SavedQuery', 'can_read'),), Pvm('SavedQueryViewApi', 'can_show'): (Pvm('SavedQuery', 'can_read'),), Pvm('SavedQueryViewApi', 'can_edit'): (Pvm('SavedQuery', 'can_write'),), Pvm('SavedQueryViewApi', 'can_list'): (Pvm('SavedQuery', 'can_read'),), Pvm('SavedQueryViewApi', 'can_add'): (Pvm('SavedQuery', 'can_write'),), Pvm('SavedQueryViewApi', 'muldelete'): (Pvm('SavedQuery', 'can_write'),)}

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