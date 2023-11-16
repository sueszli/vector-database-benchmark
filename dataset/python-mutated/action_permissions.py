from enum import Enum
from typing import Optional
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...types.uid import UID

@serializable()
class ActionPermission(Enum):
    OWNER = 1
    READ = 2
    ALL_READ = 4
    WRITE = 8
    ALL_WRITE = 32
    EXECUTE = 64
    ALL_EXECUTE = 128
COMPOUND_ACTION_PERMISSION = {ActionPermission.ALL_READ, ActionPermission.ALL_WRITE, ActionPermission.ALL_EXECUTE}

@serializable()
class ActionObjectPermission:

    def __init__(self, uid: UID, permission: ActionPermission, credentials: Optional[SyftVerifyKey]=None):
        if False:
            for i in range(10):
                print('nop')
        if credentials is None:
            if permission not in COMPOUND_ACTION_PERMISSION:
                raise Exception(f'{permission} not in {COMPOUND_ACTION_PERMISSION}')
        self.uid = uid
        self.credentials = credentials
        self.permission = permission

    @property
    def permission_string(self) -> str:
        if False:
            i = 10
            return i + 15
        if self.permission in COMPOUND_ACTION_PERMISSION:
            return f'{self.permission.name}'
        else:
            return f'{self.credentials.verify}_{self.permission.name}'

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        if self.credentials is not None:
            return f'[{self.permission.name}: {self.uid} as {self.credentials.verify}]'
        else:
            return self.permission_string

class ActionObjectOWNER(ActionObjectPermission):

    def __init__(self, uid: UID, credentials: SyftVerifyKey):
        if False:
            return 10
        self.uid = uid
        self.credentials = credentials
        self.permission = ActionPermission.OWNER

class ActionObjectREAD(ActionObjectPermission):

    def __init__(self, uid: UID, credentials: SyftVerifyKey):
        if False:
            print('Hello World!')
        self.uid = uid
        self.credentials = credentials
        self.permission = ActionPermission.READ

class ActionObjectWRITE(ActionObjectPermission):

    def __init__(self, uid: UID, credentials: SyftVerifyKey):
        if False:
            return 10
        self.uid = uid
        self.credentials = credentials
        self.permission = ActionPermission.WRITE

class ActionObjectEXECUTE(ActionObjectPermission):

    def __init__(self, uid: UID, credentials: SyftVerifyKey):
        if False:
            while True:
                i = 10
        self.uid = uid
        self.credentials = credentials
        self.permission = ActionPermission.EXECUTE