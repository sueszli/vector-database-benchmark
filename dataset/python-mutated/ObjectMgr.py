"""
Defines ObjectMgr
"""
from .ObjectMgrBase import ObjectMgrBase

class ObjectMgr(ObjectMgrBase):
    """ ObjectMgr will create, manage, update objects in the scene """

    def __init__(self, editor):
        if False:
            while True:
                i = 10
        ObjectMgrBase.__init__(self, editor)