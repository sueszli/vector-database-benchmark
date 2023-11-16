from enum import Enum
from ..types.uid import UID
from .serializable import serializable

@serializable()
class CMPCRUDPermission(Enum):
    NONE_EXECUTE = 1
    ALL_EXECUTE = 2

@serializable()
class CMPPermission:

    @property
    def permissions_string(self) -> str:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.permission_string

@serializable()
class CMPUserPermission(CMPPermission):

    def __init__(self, user_id: UID, permission: CMPCRUDPermission):
        if False:
            print('Hello World!')
        self.user_id = user_id
        self.permissions = permission

    @property
    def permission_string(self) -> str:
        if False:
            print('Hello World!')
        return f'<{self.user_uid}>_{self.permission}'

    def __repr__(self) -> str:
        if False:
            return 10
        return self.permission_string

@serializable()
class CMPCompoundPermission(CMPPermission):

    def __init__(self, permission: CMPCRUDPermission):
        if False:
            return 10
        self.permissions = permission

    @property
    def permission_string(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.permissions.name

    def __repr__(self) -> str:
        if False:
            return 10
        return self.permission_string
ALL_EXECUTE = CMPCompoundPermission(CMPCRUDPermission.ALL_EXECUTE)
NONE_EXECUTE = CMPCompoundPermission(CMPCRUDPermission.NONE_EXECUTE)