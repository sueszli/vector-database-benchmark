import logging
from dataclasses import dataclass
from sqlalchemy import Column, ForeignKey, Integer, Sequence, String, Table, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Load, relationship, Session
logger = logging.getLogger(__name__)
Base = declarative_base()

@dataclass(frozen=True)
class Pvm:
    view: str
    permission: str
PvmMigrationMapType = dict[Pvm, tuple[Pvm, ...]]

class Permission(Base):
    __tablename__ = 'ab_permission'
    id = Column(Integer, Sequence('ab_permission_id_seq'), primary_key=True)
    name = Column(String(100), unique=True, nullable=False)

    def __repr__(self) -> str:
        if False:
            return 10
        return f'{self.name}'

class ViewMenu(Base):
    __tablename__ = 'ab_view_menu'
    id = Column(Integer, Sequence('ab_view_menu_id_seq'), primary_key=True)
    name = Column(String(250), unique=True, nullable=False)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'{self.name}'

    def __eq__(self, other: object) -> bool:
        if False:
            print('Hello World!')
        return isinstance(other, self.__class__) and self.name == other.name

    def __neq__(self, other: object) -> bool:
        if False:
            return 10
        return isinstance(other, self.__class__) and self.name != other.name
assoc_permissionview_role = Table('ab_permission_view_role', Base.metadata, Column('id', Integer, Sequence('ab_permission_view_role_id_seq'), primary_key=True), Column('permission_view_id', Integer, ForeignKey('ab_permission_view.id')), Column('role_id', Integer, ForeignKey('ab_role.id')), UniqueConstraint('permission_view_id', 'role_id'))

class Role(Base):
    __tablename__ = 'ab_role'
    id = Column(Integer, Sequence('ab_role_id_seq'), primary_key=True)
    name = Column(String(64), unique=True, nullable=False)
    permissions = relationship('PermissionView', secondary=assoc_permissionview_role, backref='role')

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'{self.name}'

class PermissionView(Base):
    __tablename__ = 'ab_permission_view'
    __table_args__ = (UniqueConstraint('permission_id', 'view_menu_id'),)
    id = Column(Integer, Sequence('ab_permission_view_id_seq'), primary_key=True)
    permission_id = Column(Integer, ForeignKey('ab_permission.id'))
    permission = relationship('Permission')
    view_menu_id = Column(Integer, ForeignKey('ab_view_menu.id'))
    view_menu = relationship('ViewMenu')

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'{self.permission} {self.view_menu}'

def _add_view_menu(session: Session, view_name: str) -> ViewMenu:
    if False:
        for i in range(10):
            print('nop')
    '\n    Check and add the new view menu\n    '
    new_view = session.query(ViewMenu).filter(ViewMenu.name == view_name).one_or_none()
    if not new_view:
        new_view = ViewMenu(name=view_name)
        session.add(new_view)
    return new_view

def _add_permission(session: Session, permission_name: str) -> Permission:
    if False:
        while True:
            i = 10
    '\n    Check and add the new Permission\n    '
    new_permission = session.query(Permission).filter(Permission.name == permission_name).one_or_none()
    if not new_permission:
        new_permission = Permission(name=permission_name)
        session.add(new_permission)
    return new_permission

def _add_permission_view(session: Session, permission: Permission, view_menu: ViewMenu) -> PermissionView:
    if False:
        print('Hello World!')
    '\n    Check and add the new Permission View\n    '
    new_pvm = session.query(PermissionView).filter(PermissionView.view_menu_id == view_menu.id, PermissionView.permission_id == permission.id).one_or_none()
    if not new_pvm:
        new_pvm = PermissionView(view_menu=view_menu, permission=permission)
        session.add(new_pvm)
    return new_pvm

def _find_pvm(session: Session, view_name: str, permission_name: str) -> PermissionView:
    if False:
        for i in range(10):
            print('nop')
    return session.query(PermissionView).join(Permission).join(ViewMenu).filter(ViewMenu.name == view_name, Permission.name == permission_name).one_or_none()

def add_pvms(session: Session, pvm_data: dict[str, tuple[str, ...]], commit: bool=False) -> list[PermissionView]:
    if False:
        print('Hello World!')
    "\n    Checks if exists and adds new Permissions, Views and PermissionView's\n    "
    pvms = []
    for (view_name, permissions) in pvm_data.items():
        new_view = _add_view_menu(session, view_name)
        for permission_name in permissions:
            new_permission = _add_permission(session, permission_name)
            pvms.append(_add_permission_view(session, new_permission, new_view))
    if commit:
        session.commit()
    return pvms

def _delete_old_permissions(session: Session, pvm_map: dict[PermissionView, list[PermissionView]]) -> None:
    if False:
        i = 10
        return i + 15
    "\n    Delete old permissions:\n    - Delete the PermissionView\n    - Deletes the Permission if it's an orphan now\n    - Deletes the ViewMenu if it's an orphan now\n    "
    for (old_pvm, new_pvms) in pvm_map.items():
        old_permission_name = old_pvm.permission.name
        old_view_name = old_pvm.view_menu.name
        logger.info(f'Going to delete pvm: {old_pvm}')
        session.delete(old_pvm)
        pvms_with_permission = session.query(PermissionView).join(Permission).filter(Permission.name == old_permission_name).first()
        if not pvms_with_permission:
            logger.info(f'Going to delete permission: {old_pvm.permission}')
            session.delete(old_pvm.permission)
        pvms_with_view_menu = session.query(PermissionView).join(ViewMenu).filter(ViewMenu.name == old_view_name).first()
        if not pvms_with_view_menu:
            logger.info(f'Going to delete view_menu: {old_pvm.view_menu}')
            session.delete(old_pvm.view_menu)

def migrate_roles(session: Session, pvm_key_map: PvmMigrationMapType, commit: bool=False) -> None:
    if False:
        print('Hello World!')
    '\n    Migrates all existing roles that have the permissions to be migrated\n    '
    pvm_map: dict[PermissionView, list[PermissionView]] = {}
    for (old_pvm_key, new_pvms_) in pvm_key_map.items():
        old_pvm = _find_pvm(session, old_pvm_key.view, old_pvm_key.permission)
        if old_pvm:
            for new_pvm_key in new_pvms_:
                new_pvm = _find_pvm(session, new_pvm_key.view, new_pvm_key.permission)
                if old_pvm not in pvm_map:
                    pvm_map[old_pvm] = [new_pvm]
                else:
                    pvm_map[old_pvm].append(new_pvm)
    roles = session.query(Role).options(Load(Role).joinedload(Role.permissions)).all()
    for role in roles:
        for (old_pvm, new_pvms) in pvm_map.items():
            if old_pvm in role.permissions:
                logger.info(f'Removing {old_pvm} from {role}')
                role.permissions.remove(old_pvm)
                for new_pvm in new_pvms:
                    if new_pvm not in role.permissions:
                        logger.info(f'Add {new_pvm} to {role}')
                        role.permissions.append(new_pvm)
        session.merge(role)
    _delete_old_permissions(session, pvm_map)
    if commit:
        session.commit()

def get_reversed_new_pvms(pvm_map: PvmMigrationMapType) -> dict[str, tuple[str, ...]]:
    if False:
        return 10
    reversed_pvms: dict[str, tuple[str, ...]] = {}
    for (old_pvm, new_pvms) in pvm_map.items():
        if old_pvm.view not in reversed_pvms:
            reversed_pvms[old_pvm.view] = (old_pvm.permission,)
        else:
            reversed_pvms[old_pvm.view] = reversed_pvms[old_pvm.view] + (old_pvm.permission,)
    return reversed_pvms

def get_reversed_pvm_map(pvm_map: PvmMigrationMapType) -> PvmMigrationMapType:
    if False:
        while True:
            i = 10
    reversed_pvm_map: PvmMigrationMapType = {}
    for (old_pvm, new_pvms) in pvm_map.items():
        for new_pvm in new_pvms:
            if new_pvm not in reversed_pvm_map:
                reversed_pvm_map[new_pvm] = (old_pvm,)
            else:
                reversed_pvm_map[new_pvm] = reversed_pvm_map[new_pvm] + (old_pvm,)
    return reversed_pvm_map