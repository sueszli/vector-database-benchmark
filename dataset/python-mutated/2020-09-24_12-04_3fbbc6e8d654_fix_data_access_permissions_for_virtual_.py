"""fix data access permissions for virtual datasets

Revision ID: 3fbbc6e8d654
Revises: e5ef6828ac4e
Create Date: 2020-09-24 12:04:33.827436

"""
revision = '3fbbc6e8d654'
down_revision = 'e5ef6828ac4e'
import re
from alembic import op
from sqlalchemy import Column, ForeignKey, Integer, orm, Sequence, String, Table, UniqueConstraint
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, relationship
Base = declarative_base()

class Permission(Base):
    __tablename__ = 'ab_permission'
    id = Column(Integer, Sequence('ab_permission_id_seq'), primary_key=True)
    name = Column(String(100), unique=True, nullable=False)

class ViewMenu(Base):
    __tablename__ = 'ab_view_menu'
    id = Column(Integer, Sequence('ab_view_menu_id_seq'), primary_key=True)
    name = Column(String(250), unique=True, nullable=False)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, self.__class__) and self.name == other.name

    def __neq__(self, other):
        if False:
            print('Hello World!')
        return self.name != other.name
assoc_permissionview_role = Table('ab_permission_view_role', Base.metadata, Column('id', Integer, Sequence('ab_permission_view_role_id_seq'), primary_key=True), Column('permission_view_id', Integer, ForeignKey('ab_permission_view.id')), Column('role_id', Integer, ForeignKey('ab_role.id')), UniqueConstraint('permission_view_id', 'role_id'))

class Role(Base):
    __tablename__ = 'ab_role'
    id = Column(Integer, Sequence('ab_role_id_seq'), primary_key=True)
    name = Column(String(64), unique=True, nullable=False)
    permissions = relationship('PermissionView', secondary=assoc_permissionview_role, backref='role')

class PermissionView(Base):
    __tablename__ = 'ab_permission_view'
    __table_args__ = (UniqueConstraint('permission_id', 'view_menu_id'),)
    id = Column(Integer, Sequence('ab_permission_view_id_seq'), primary_key=True)
    permission_id = Column(Integer, ForeignKey('ab_permission.id'))
    permission = relationship('Permission')
    view_menu_id = Column(Integer, ForeignKey('ab_view_menu.id'))
    view_menu = relationship('ViewMenu')
sqlatable_user = Table('sqlatable_user', Base.metadata, Column('id', Integer, primary_key=True), Column('user_id', Integer, ForeignKey('ab_user.id')), Column('table_id', Integer, ForeignKey('tables.id')))

class Database(Base):
    """An ORM object that stores Database related information"""
    __tablename__ = 'dbs'
    __table_args__ = (UniqueConstraint('database_name'),)
    id = Column(Integer, primary_key=True)
    verbose_name = Column(String(250), unique=True)
    database_name = Column(String(250), unique=True, nullable=False)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return self.name

    @property
    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.verbose_name if self.verbose_name else self.database_name

class SqlaTable(Base):
    __tablename__ = 'tables'
    __table_args__ = (UniqueConstraint('database_id', 'table_name'),)
    id = Column(Integer, primary_key=True)
    table_name = Column(String(250), nullable=False)
    database_id = Column(Integer, ForeignKey('dbs.id'), nullable=False)
    database = relationship('Database', backref=backref('tables', cascade='all, delete-orphan'), foreign_keys=[database_id])

    def get_perm(self) -> str:
        if False:
            while True:
                i = 10
        return f'[{self.database}].[{self.table_name}](id:{self.id})'

def upgrade():
    if False:
        i = 10
        return i + 15
    '\n    Previous sqla_viz behaviour when creating a virtual dataset was faulty\n    by creating an associated data access permission with [None] on the database name.\n\n    This migration revision, fixes all faulty permissions that may exist on the db\n    Only fixes permissions that still have an associated dataset (fetch by id)\n    and replaces them with the current (correct) permission name\n    '
    bind = op.get_bind()
    session = orm.Session(bind=bind)
    faulty_view_menus = session.query(ViewMenu).join(PermissionView).join(Permission).filter(ViewMenu.name.ilike('[None].[%](id:%)')).filter(Permission.name == 'datasource_access').all()
    orphaned_faulty_view_menus = []
    for faulty_view_menu in faulty_view_menus:
        match_ds_id = re.match('\\[None\\]\\.\\[.*\\]\\(id:(\\d+)\\)', faulty_view_menu.name)
        if match_ds_id:
            dataset_id = int(match_ds_id.group(1))
            dataset = session.query(SqlaTable).get(dataset_id)
            if dataset:
                try:
                    new_view_menu = dataset.get_perm()
                except Exception:
                    return
                existing_view_menu = session.query(ViewMenu).filter(ViewMenu.name == new_view_menu).one_or_none()
                if existing_view_menu:
                    orphaned_faulty_view_menus.append(faulty_view_menu)
                else:
                    faulty_view_menu.name = new_view_menu
    try:
        session.commit()
    except SQLAlchemyError:
        session.rollback()
    for orphaned_faulty_view_menu in orphaned_faulty_view_menus:
        pvm = session.query(PermissionView).filter(PermissionView.view_menu == orphaned_faulty_view_menu).one_or_none()
        if pvm:
            roles = session.query(Role).filter(Role.permissions.contains(pvm)).all()
            for role in roles:
                if pvm in role.permissions:
                    role.permissions.remove(pvm)
            session.delete(pvm)
        session.delete(orphaned_faulty_view_menu)
    try:
        session.commit()
    except SQLAlchemyError:
        session.rollback()

def downgrade():
    if False:
        return 10
    pass