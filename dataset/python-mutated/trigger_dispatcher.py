from __future__ import absolute_import
import six
from oslo_config import cfg
from jsonschema import ValidationError
from st2common.models.api.trace import TraceContext
from st2common.transport.reactor import TriggerDispatcher
from st2common.validators.api.reactor import validate_trigger_payload
__all__ = ['TriggerDispatcherService']

class TriggerDispatcherService(object):
    """
    Class for handling dispatching of trigger.
    """

    def __init__(self, logger):
        if False:
            return 10
        self._logger = logger
        self._dispatcher = TriggerDispatcher(self._logger)

    def dispatch(self, trigger, payload=None, trace_tag=None, throw_on_validation_error=False):
        if False:
            return 10
        '\n        Method which dispatches the trigger.\n\n        :param trigger: Reference to the TriggerTypeDB (<pack>.<name>) or TriggerDB object.\n        :type trigger: ``str``\n\n        :param payload: Trigger payload.\n        :type payload: ``dict``\n\n        :param trace_tag: Tracer to track the triggerinstance.\n        :type trace_tags: ``str``\n\n        :param throw_on_validation_error: True to throw on validation error (if validate_payload is\n                                          True) instead of logging the error.\n        :type throw_on_validation_error: ``boolean``\n        '
        trace_context = TraceContext(trace_tag=trace_tag) if trace_tag else None
        self._logger.debug('Added trace_context %s to trigger %s.', trace_context, trigger)
        return self.dispatch_with_context(trigger, payload=payload, trace_context=trace_context, throw_on_validation_error=throw_on_validation_error)

    def dispatch_with_context(self, trigger, payload=None, trace_context=None, throw_on_validation_error=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Method which dispatches the trigger.\n\n        :param trigger: Reference to the TriggerTypeDB (<pack>.<name>) or TriggerDB object.\n        :type trigger: ``str``\n\n        :param payload: Trigger payload.\n        :type payload: ``dict``\n\n        :param trace_context: Trace context to associate with Trigger.\n        :type trace_context: ``st2common.api.models.api.trace.TraceContext``\n\n        :param throw_on_validation_error: True to throw on validation error (if validate_payload is\n                                          True) instead of logging the error.\n        :type throw_on_validation_error: ``boolean``\n        '
        try:
            validate_trigger_payload(trigger_type_ref=trigger, payload=payload, throw_on_inexistent_trigger=True)
        except (ValidationError, ValueError, Exception) as e:
            self._logger.warn('Failed to validate payload (%s) for trigger "%s": %s' % (str(payload), trigger, six.text_type(e)))
            if cfg.CONF.system.validate_trigger_payload:
                msg = 'Trigger payload validation failed and validation is enabled, not dispatching a trigger "%s" (%s): %s' % (trigger, str(payload), six.text_type(e))
                if throw_on_validation_error:
                    raise ValueError(msg)
                self._logger.warn(msg)
                return None
        self._logger.debug('Dispatching trigger %s with payload %s.', trigger, payload)
        return self._dispatcher.dispatch(trigger, payload=payload, trace_context=trace_context)