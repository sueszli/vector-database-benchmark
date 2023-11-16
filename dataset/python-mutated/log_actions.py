import uuid
from asgiref.local import Local
from django.utils.functional import LazyObject
from wagtail import hooks
from wagtail.utils.registry import ObjectTypeRegistry

class LogFormatter:
    """
    Defines how to format log messages / comments for a particular action type. Messages that depend on
    log entry data should override format_message / format_comment; static messages can just be set as the
    'message' / 'comment' attribute.

    To be registered with log_registry.register_action.
    """
    label = ''
    message = ''
    comment = ''

    def format_message(self, log_entry):
        if False:
            print('Hello World!')
        return self.message

    def format_comment(self, log_entry):
        if False:
            for i in range(10):
                print('nop')
        return self.comment
_active = Local()

class LogContext:
    """
    Stores data about the environment in which a logged action happens -
    e.g. the active user - to be stored in the log entry for that action.
    """

    def __init__(self, user=None, generate_uuid=True):
        if False:
            i = 10
            return i + 15
        self.user = user
        if generate_uuid:
            self.uuid = uuid.uuid4()
        else:
            self.uuid = None

    def __enter__(self):
        if False:
            print('Hello World!')
        self._old_log_context = getattr(_active, 'value', None)
        activate(self)
        return self

    def __exit__(self, type, value, traceback):
        if False:
            while True:
                i = 10
        if self._old_log_context:
            activate(self._old_log_context)
        else:
            deactivate()
empty_log_context = LogContext(generate_uuid=False)

def activate(log_context):
    if False:
        return 10
    _active.value = log_context

def deactivate():
    if False:
        for i in range(10):
            print('nop')
    del _active.value

def get_active_log_context():
    if False:
        for i in range(10):
            print('nop')
    return getattr(_active, 'value', empty_log_context)

class LogActionRegistry:
    """
    A central store for log actions.
    The expected format for registered log actions: Namespaced action, Action label, Action message (or callable)
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.has_scanned_for_actions = False
        self.formatters = {}
        self.choices = []
        self.log_entry_models_by_type = ObjectTypeRegistry()
        self.log_entry_models = set()

    def scan_for_actions(self):
        if False:
            while True:
                i = 10
        if not self.has_scanned_for_actions:
            for fn in hooks.get_hooks('register_log_actions'):
                fn(self)
            self.has_scanned_for_actions = True

    def register_model(self, cls, log_entry_model):
        if False:
            i = 10
            return i + 15
        self.log_entry_models_by_type.register(cls, value=log_entry_model)
        self.log_entry_models.add(log_entry_model)

    def register_action(self, action, *args):
        if False:
            return 10

        def register_formatter_class(formatter_cls):
            if False:
                print('Hello World!')
            formatter = formatter_cls()
            self.formatters[action] = formatter
            self.choices.append((action, formatter.label))
        if args:
            (label, message) = args
            formatter_cls = type('_LogFormatter', (LogFormatter,), {'label': label, 'message': message})
            register_formatter_class(formatter_cls)
        else:
            return register_formatter_class

    def get_choices(self):
        if False:
            return 10
        self.scan_for_actions()
        return self.choices

    def get_formatter(self, log_entry):
        if False:
            i = 10
            return i + 15
        self.scan_for_actions()
        return self.formatters.get(log_entry.action)

    def action_exists(self, action):
        if False:
            return 10
        self.scan_for_actions()
        return action in self.formatters

    def get_log_entry_models(self):
        if False:
            return 10
        self.scan_for_actions()
        return self.log_entry_models

    def get_action_label(self, action):
        if False:
            for i in range(10):
                print('nop')
        return self.formatters[action].label

    def get_log_model_for_model(self, model):
        if False:
            i = 10
            return i + 15
        self.scan_for_actions()
        return self.log_entry_models_by_type.get_by_type(model)

    def get_log_model_for_instance(self, instance):
        if False:
            print('Hello World!')
        if isinstance(instance, LazyObject):
            instance = instance._wrapped
        return self.get_log_model_for_model(type(instance))

    def log(self, instance, action, user=None, uuid=None, **kwargs):
        if False:
            print('Hello World!')
        self.scan_for_actions()
        log_entry_model = self.get_log_model_for_instance(instance)
        if log_entry_model is None:
            return
        user = user or get_active_log_context().user
        uuid = uuid or get_active_log_context().uuid
        return log_entry_model.objects.log_action(instance, action, user=user, uuid=uuid, **kwargs)

    def get_logs_for_instance(self, instance):
        if False:
            i = 10
            return i + 15
        log_entry_model = self.get_log_model_for_instance(instance)
        if log_entry_model is None:
            from wagtail.models import ModelLogEntry
            return ModelLogEntry.objects.none()
        return log_entry_model.objects.for_instance(instance)
registry = LogActionRegistry()

def log(instance, action, **kwargs):
    if False:
        return 10
    return registry.log(instance, action, **kwargs)