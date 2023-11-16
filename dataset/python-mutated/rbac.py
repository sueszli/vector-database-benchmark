from __future__ import absolute_import
from st2common.persistence import base
from st2common.models.db.rbac import role_access
from st2common.models.db.rbac import user_role_assignment_access
from st2common.models.db.rbac import permission_grant_access
from st2common.models.db.rbac import group_to_role_mapping_access
__all__ = ['Role', 'UserRoleAssignment', 'PermissionGrant', 'GroupToRoleMapping']

class Role(base.Access):
    impl = role_access

    @classmethod
    def _get_impl(cls):
        if False:
            while True:
                i = 10
        return cls.impl

class UserRoleAssignment(base.Access):
    impl = user_role_assignment_access

    @classmethod
    def _get_impl(cls):
        if False:
            for i in range(10):
                print('nop')
        return cls.impl

class PermissionGrant(base.Access):
    impl = permission_grant_access

    @classmethod
    def _get_impl(cls):
        if False:
            print('Hello World!')
        return cls.impl

class GroupToRoleMapping(base.Access):
    impl = group_to_role_mapping_access

    @classmethod
    def _get_impl(cls):
        if False:
            i = 10
            return i + 15
        return cls.impl