"""security converge css templates

Revision ID: 8ee129739cf9
Revises: e38177dbf641
Create Date: 2020-11-30 17:54:09.118630

"""
revision = '8ee129739cf9'
down_revision = 'e38177dbf641'
from alembic import op
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from superset.migrations.shared.security_converge import add_pvms, get_reversed_new_pvms, get_reversed_pvm_map, migrate_roles, Pvm
NEW_PVMS = {'CssTemplate': ('can_read', 'can_write')}
PVM_MAP = {Pvm('CssTemplateModelView', 'can_list'): (Pvm('CssTemplate', 'can_read'),), Pvm('CssTemplateModelView', 'can_show'): (Pvm('CssTemplate', 'can_read'),), Pvm('CssTemplateModelView', 'can_add'): (Pvm('CssTemplate', 'can_write'),), Pvm('CssTemplateModelView', 'can_edit'): (Pvm('CssTemplate', 'can_write'),), Pvm('CssTemplateModelView', 'can_delete'): (Pvm('CssTemplate', 'can_write'),), Pvm('CssTemplateModelView', 'muldelete'): (Pvm('CssTemplate', 'can_write'),), Pvm('CssTemplateAsyncModelView', 'can_list'): (Pvm('CssTemplate', 'can_read'),), Pvm('CssTemplateAsyncModelView', 'muldelete'): (Pvm('CssTemplate', 'can_write'),)}

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