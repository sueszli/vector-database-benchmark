"""The fallback skill implements a special type of skill handling
utterances not handled by the intent system.
"""
import operator
from mycroft.metrics import report_timing, Stopwatch
from mycroft.util.log import LOG
from .mycroft_skill import MycroftSkill, get_handler_name

class FallbackSkill(MycroftSkill):
    """Fallbacks come into play when no skill matches an Adapt or closely with
    a Padatious intent.  All Fallback skills work together to give them a
    view of the user's utterance.  Fallback handlers are called in an order
    determined the priority provided when the the handler is registered.

    ========   ========   ================================================
    Priority   Who?       Purpose
    ========   ========   ================================================
       1-4     RESERVED   Unused for now, slot for pre-Padatious if needed
         5     MYCROFT    Padatious near match (conf > 0.8)
      6-88     USER       General
        89     MYCROFT    Padatious loose match (conf > 0.5)
     90-99     USER       Uncaught intents
       100+    MYCROFT    Fallback Unknown or other future use
    ========   ========   ================================================

    Handlers with the numerically lowest priority are invoked first.
    Multiple fallbacks can exist at the same priority, but no order is
    guaranteed.

    A Fallback can either observe or consume an utterance. A consumed
    utterance will not be see by any other Fallback handlers.
    """
    fallback_handlers = {}
    wrapper_map = []

    def __init__(self, name=None, bus=None, use_settings=True):
        if False:
            print('Hello World!')
        super().__init__(name, bus, use_settings)
        self.instance_fallback_handlers = []

    @classmethod
    def make_intent_failure_handler(cls, bus):
        if False:
            i = 10
            return i + 15
        'Goes through all fallback handlers until one returns True'

        def handler(message):
            if False:
                while True:
                    i = 10
            (start, stop) = message.data.get('fallback_range', (0, 101))
            LOG.debug('Checking fallbacks in range {} - {}'.format(start, stop))
            bus.emit(message.forward('mycroft.skill.handler.start', data={'handler': 'fallback'}))
            stopwatch = Stopwatch()
            handler_name = None
            with stopwatch:
                sorted_handlers = sorted(cls.fallback_handlers.items(), key=operator.itemgetter(0))
                handlers = [f[1] for f in sorted_handlers if start <= f[0] < stop]
                for handler in handlers:
                    try:
                        if handler(message):
                            status = True
                            handler_name = get_handler_name(handler)
                            bus.emit(message.forward('mycroft.skill.handler.complete', data={'handler': 'fallback', 'fallback_handler': handler_name}))
                            break
                    except Exception:
                        LOG.exception('Exception in fallback.')
                else:
                    status = False
                    warning = 'No fallback could handle intent.'
                    bus.emit(message.forward('mycroft.skill.handler.complete', data={'handler': 'fallback', 'exception': warning}))
            bus.emit(message.response(data={'handled': status}))
            if message.context.get('ident'):
                ident = message.context['ident']
                report_timing(ident, 'fallback_handler', stopwatch, {'handler': handler_name})
        return handler

    @classmethod
    def _register_fallback(cls, handler, wrapper, priority):
        if False:
            i = 10
            return i + 15
        'Register a function to be called as a general info fallback\n        Fallback should receive message and return\n        a boolean (True if succeeded or False if failed)\n\n        Lower priority gets run first\n        0 for high priority 100 for low priority\n\n        Args:\n            handler (callable): original handler, used as a reference when\n                                removing\n            wrapper (callable): wrapped version of handler\n            priority (int): fallback priority\n        '
        while priority in cls.fallback_handlers:
            priority += 1
        cls.fallback_handlers[priority] = wrapper
        cls.wrapper_map.append((handler, wrapper))

    def register_fallback(self, handler, priority):
        if False:
            i = 10
            return i + 15
        'Register a fallback with the list of fallback handlers and with the\n        list of handlers registered by this instance\n        '

        def wrapper(*args, **kwargs):
            if False:
                while True:
                    i = 10
            if handler(*args, **kwargs):
                self.make_active()
                return True
            return False
        self.instance_fallback_handlers.append(handler)
        self._register_fallback(handler, wrapper, priority)

    @classmethod
    def _remove_registered_handler(cls, wrapper_to_del):
        if False:
            while True:
                i = 10
        'Remove a registered wrapper.\n\n        Args:\n            wrapper_to_del (callable): wrapped handler to be removed\n\n        Returns:\n            (bool) True if one or more handlers were removed, otherwise False.\n        '
        found_handler = False
        for (priority, handler) in list(cls.fallback_handlers.items()):
            if handler == wrapper_to_del:
                found_handler = True
                del cls.fallback_handlers[priority]
        if not found_handler:
            LOG.warning('No fallback matching {}'.format(wrapper_to_del))
        return found_handler

    @classmethod
    def remove_fallback(cls, handler_to_del):
        if False:
            return 10
        'Remove a fallback handler.\n\n        Args:\n            handler_to_del: reference to handler\n        Returns:\n            (bool) True if at least one handler was removed, otherwise False\n        '
        wrapper_to_del = None
        for (h, w) in cls.wrapper_map:
            if handler_to_del in (h, w):
                wrapper_to_del = w
                break
        if wrapper_to_del:
            cls.wrapper_map.remove((h, w))
            remove_ok = cls._remove_registered_handler(wrapper_to_del)
        else:
            LOG.warning('Could not find matching fallback handler')
            remove_ok = False
        return remove_ok

    def remove_instance_handlers(self):
        if False:
            while True:
                i = 10
        'Remove all fallback handlers registered by the fallback skill.'
        self.log.info('Removing all handlers...')
        while len(self.instance_fallback_handlers):
            handler = self.instance_fallback_handlers.pop()
            self.remove_fallback(handler)

    def default_shutdown(self):
        if False:
            i = 10
            return i + 15
        'Remove all registered handlers and perform skill shutdown.'
        self.remove_instance_handlers()
        super(FallbackSkill, self).default_shutdown()