from collections import defaultdict
from superset import security_manager

def cleanup_permissions() -> None:
    if False:
        print('Hello World!')
    pvms = security_manager.get_session.query(security_manager.permissionview_model).all()
    print(f'# of permission view menus is: {len(pvms)}')
    pvms_dict = defaultdict(list)
    for pvm in pvms:
        pvms_dict[pvm.permission, pvm.view_menu].append(pvm)
    duplicates = [v for v in pvms_dict.values() if len(v) > 1]
    len(duplicates)
    for pvm_list in duplicates:
        first_prm = pvm_list[0]
        roles = set(first_prm.role)
        for pvm in pvm_list[1:]:
            roles = roles.union(pvm.role)
            security_manager.get_session.delete(pvm)
        first_prm.roles = list(roles)
    security_manager.get_session.commit()
    pvms = security_manager.get_session.query(security_manager.permissionview_model).all()
    print(f'Stage 1: # of permission view menus is: {len(pvms)}')
    pvms = security_manager.get_session.query(security_manager.permissionview_model).all()
    for pvm in pvms:
        if not (pvm.view_menu and pvm.permission):
            security_manager.get_session.delete(pvm)
    security_manager.get_session.commit()
    pvms = security_manager.get_session.query(security_manager.permissionview_model).all()
    print(f'Stage 2: # of permission view menus is: {len(pvms)}')
    roles = security_manager.get_session.query(security_manager.role_model).all()
    for role in roles:
        role.permissions = [p for p in role.permissions if p]
    security_manager.get_session.commit()
    pvms = security_manager.get_session.query(security_manager.permissionview_model).all()
    for pvm in pvms:
        pvm.role = [r for r in pvm.role if r]
    security_manager.get_session.commit()
cleanup_permissions()