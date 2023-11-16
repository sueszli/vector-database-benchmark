"""
Defines AnimMgr
"""
from .AnimMgrBase import AnimMgrBase

class AnimMgr(AnimMgrBase):
    """ Animation will create, manage, update animations in the scene """

    def __init__(self, editor):
        if False:
            return 10
        AnimMgrBase.__init__(self, editor)