from __future__ import absolute_import
from st2common import transport
from st2common.models.db.executionstate import actionexecstate_access
from st2common.persistence import base as persistence
__all__ = ['ActionExecutionState']

class ActionExecutionState(persistence.Access):
    impl = actionexecstate_access
    publisher = None

    @classmethod
    def _get_impl(cls):
        if False:
            return 10
        return cls.impl

    @classmethod
    def _get_publisher(cls):
        if False:
            return 10
        if not cls.publisher:
            cls.publisher = transport.actionexecutionstate.ActionExecutionStatePublisher()
        return cls.publisher