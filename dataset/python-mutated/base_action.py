from __future__ import absolute_import
import abc
import six
from st2common.runners.utils import get_logger_for_python_runner_action
from st2common.runners.utils import PackConfigDict
__all__ = ['Action']

@six.add_metaclass(abc.ABCMeta)
class Action(object):
    """
    Base action class other Python actions should inherit from.
    """
    description = None

    def __init__(self, config=None, action_service=None):
        if False:
            i = 10
            return i + 15
        '\n        :param config: Action config.\n        :type config: ``dict``\n\n        :param action_service: ActionService object.\n        :type action_service: :class:`ActionService~\n        '
        self.config = config or {}
        self.action_service = action_service
        if action_service and getattr(action_service, '_action_wrapper', None):
            log_level = getattr(action_service._action_wrapper, '_log_level', 'debug')
            pack_name = getattr(action_service._action_wrapper, '_pack', 'unknown')
        else:
            log_level = 'debug'
            pack_name = 'unknown'
        self.config = PackConfigDict(pack_name, self.config)
        self.logger = get_logger_for_python_runner_action(action_name=self.__class__.__name__, log_level=log_level)

    @abc.abstractmethod
    def run(self, **kwargs):
        if False:
            i = 10
            return i + 15
        pass