"""security converge charts

Revision ID: ccb74baaa89b
Revises: 811494c0cc23
Create Date: 2020-12-09 14:13:48.058003

"""
revision = 'ccb74baaa89b'
down_revision = '40f16acf1ba7'
from alembic import op
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from superset.migrations.shared.security_converge import add_pvms, get_reversed_new_pvms, get_reversed_pvm_map, migrate_roles, Pvm
NEW_PVMS = {'Chart': ('can_read', 'can_write')}
PVM_MAP = {Pvm('SliceModelView', 'can_list'): (Pvm('Chart', 'can_read'),), Pvm('SliceModelView', 'can_show'): (Pvm('Chart', 'can_read'),), Pvm('SliceModelView', 'can_edit'): (Pvm('Chart', 'can_write'),), Pvm('SliceModelView', 'can_delete'): (Pvm('Chart', 'can_write'),), Pvm('SliceModelView', 'can_add'): (Pvm('Chart', 'can_write'),), Pvm('SliceModelView', 'can_download'): (Pvm('Chart', 'can_read'),), Pvm('SliceModelView', 'muldelete'): (Pvm('Chart', 'can_write'),), Pvm('SliceModelView', 'can_mulexport'): (Pvm('Chart', 'can_read'),), Pvm('SliceModelView', 'can_favorite_status'): (Pvm('Chart', 'can_read'),), Pvm('SliceModelView', 'can_cache_screenshot'): (Pvm('Chart', 'can_read'),), Pvm('SliceModelView', 'can_screenshot'): (Pvm('Chart', 'can_read'),), Pvm('SliceModelView', 'can_data_from_cache'): (Pvm('Chart', 'can_read'),), Pvm('SliceAsync', 'can_list'): (Pvm('Chart', 'can_read'),), Pvm('SliceAsync', 'muldelete'): (Pvm('Chart', 'can_write'),)}

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