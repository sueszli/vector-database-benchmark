"""Plugin mixin class for ScheduleMixin."""
import logging
from django.conf import settings
from django.db.utils import OperationalError, ProgrammingError
from plugin.helpers import MixinImplementationError
logger = logging.getLogger('inventree')

class ScheduleMixin:
    """Mixin that provides support for scheduled tasks.

    Implementing classes must provide a dict object called SCHEDULED_TASKS,
    which provides information on the tasks to be scheduled.

    SCHEDULED_TASKS = {
        # Name of the task (will be prepended with the plugin name)
        'test_server': {
            'func': 'myplugin.tasks.test_server',   # Python function to call (no arguments!)
            'schedule': "I",                        # Schedule type (see django_q.Schedule)
            'minutes': 30,                          # Number of minutes (only if schedule type = Minutes)
            'repeats': 5,                           # Number of repeats (leave blank for 'forever')
        },
        'member_func': {
            'func': 'my_class_func',                # Note, without the 'dot' notation, it will call a class member function
            'schedule': "H",                        # Once per hour
        },
    }

    Note: 'schedule' parameter must be one of ['I', 'H', 'D', 'W', 'M', 'Q', 'Y']

    Note: The 'func' argument can take two different forms:
        - Dotted notation e.g. 'module.submodule.func' - calls a global function with the defined path
        - Member notation e.g. 'my_func' (no dots!) - calls a member function of the calling class
    """
    ALLOWABLE_SCHEDULE_TYPES = ['I', 'H', 'D', 'W', 'M', 'Q', 'Y']
    SCHEDULED_TASKS = {}

    class MixinMeta:
        """Meta options for this mixin."""
        MIXIN_NAME = 'Schedule'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Register mixin.'
        super().__init__()
        self.scheduled_tasks = self.get_scheduled_tasks()
        self.validate_scheduled_tasks()
        self.add_mixin('schedule', 'has_scheduled_tasks', __class__)

    @classmethod
    def _activate_mixin(cls, registry, plugins, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Activate schedules from plugins with the ScheduleMixin.'
        logger.debug('Activating plugin tasks')
        from common.models import InvenTreeSetting
        task_keys = []
        if settings.PLUGIN_TESTING or InvenTreeSetting.get_setting('ENABLE_PLUGINS_SCHEDULE'):
            for (_key, plugin) in plugins:
                if plugin.mixin_enabled('schedule'):
                    if plugin.is_active():
                        plugin.register_tasks()
                        task_keys += plugin.get_task_names()
        if len(task_keys) > 0:
            logger.info('Activated %s scheduled tasks', len(task_keys))
        try:
            from django_q.models import Schedule
            scheduled_plugin_tasks = Schedule.objects.filter(name__istartswith='plugin.')
            deleted_count = 0
            for task in scheduled_plugin_tasks:
                if task.name not in task_keys:
                    task.delete()
                    deleted_count += 1
            if deleted_count > 0:
                logger.info('Removed %s old scheduled tasks', deleted_count)
        except (ProgrammingError, OperationalError):
            logger.warning('activate_integration_schedule failed, database not ready')

    def get_scheduled_tasks(self):
        if False:
            i = 10
            return i + 15
        'Returns `SCHEDULED_TASKS` context.\n\n        Override if you want the scheduled tasks to be dynamic (influenced by settings for example).\n        '
        return getattr(self, 'SCHEDULED_TASKS', {})

    @property
    def has_scheduled_tasks(self):
        if False:
            while True:
                i = 10
        'Are tasks defined for this plugin.'
        return bool(self.scheduled_tasks)

    def validate_scheduled_tasks(self):
        if False:
            print('Hello World!')
        'Check that the provided scheduled tasks are valid.'
        if not self.has_scheduled_tasks:
            raise MixinImplementationError('SCHEDULED_TASKS not defined')
        for (key, task) in self.scheduled_tasks.items():
            if 'func' not in task:
                raise MixinImplementationError(f"Task '{key}' is missing 'func' parameter")
            if 'schedule' not in task:
                raise MixinImplementationError(f"Task '{key}' is missing 'schedule' parameter")
            schedule = task['schedule'].upper().strip()
            if schedule not in self.ALLOWABLE_SCHEDULE_TYPES:
                raise MixinImplementationError(f"Task '{key}': Schedule '{schedule}' is not a valid option")
            if schedule == 'I' and 'minutes' not in task:
                raise MixinImplementationError(f"Task '{key}' is missing 'minutes' parameter")

    def get_task_name(self, key):
        if False:
            return 10
        'Task name for key.'
        slug = self.plugin_slug()
        return f'plugin.{slug}.{key}'

    def get_task_names(self):
        if False:
            return 10
        'All defined task names.'
        return [self.get_task_name(key) for key in self.scheduled_tasks.keys()]

    def register_tasks(self):
        if False:
            print('Hello World!')
        'Register the tasks with the database.'
        try:
            from django_q.models import Schedule
            for (key, task) in self.scheduled_tasks.items():
                task_name = self.get_task_name(key)
                obj = {'name': task_name, 'schedule_type': task['schedule'], 'minutes': task.get('minutes', None), 'repeats': task.get('repeats', -1)}
                func_name = task['func'].strip()
                if '.' in func_name:
                    'Dotted notation indicates that we wish to run a globally defined function, from a specified Python module.'
                    obj['func'] = func_name
                else:
                    "Non-dotted notation indicates that we wish to call a 'member function' of the calling plugin. This is managed by the plugin registry itself."
                    slug = self.plugin_slug()
                    obj['func'] = 'plugin.registry.call_plugin_function'
                    obj['args'] = f"'{slug}', '{func_name}'"
                if Schedule.objects.filter(name=task_name).exists():
                    logger.info("Updating scheduled task '%s'", task_name)
                    instance = Schedule.objects.get(name=task_name)
                    for item in obj:
                        setattr(instance, item, obj[item])
                    instance.save()
                else:
                    logger.info("Adding scheduled task '%s'", task_name)
                    Schedule.objects.create(**obj)
        except (ProgrammingError, OperationalError):
            logger.warning('register_tasks failed, database not ready')

    def unregister_tasks(self):
        if False:
            return 10
        'Deregister the tasks with the database.'
        try:
            from django_q.models import Schedule
            for (key, _) in self.scheduled_tasks.items():
                task_name = self.get_task_name(key)
                try:
                    scheduled_task = Schedule.objects.get(name=task_name)
                    scheduled_task.delete()
                except Schedule.DoesNotExist:
                    pass
        except (ProgrammingError, OperationalError):
            logger.warning('unregister_tasks failed, database not ready')