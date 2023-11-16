from functools import wraps
from django.conf import settings
from . import tasks

def _sse_enabled(get_project_from_first_arg=lambda obj: obj.project):
    if False:
        return 10
    '\n    Decorator that only call the service function if sse is enabled else return None.\n    i.e: settings are configured and the project has sse enabled.\n\n    :param get_project_from_first_arg: function that takes the first argument\n        of the decorated function and returns the project object.\n    '

    def decorator(service_func):
        if False:
            while True:
                i = 10

        @wraps(service_func)
        def wrapper(*args, **kwargs):
            if False:
                return 10
            project = get_project_from_first_arg(args[0])
            if all([settings.SSE_SERVER_BASE_URL, settings.SSE_AUTHENTICATION_TOKEN, project.enable_realtime_updates]):
                return service_func(*args, **kwargs)
            return None
        return wrapper
    return decorator

@_sse_enabled(get_project_from_first_arg=lambda obj: obj)
def send_environment_update_message_for_project(project):
    if False:
        while True:
            i = 10
    tasks.send_environment_update_message_for_project.delay(args=(project.id,))

@_sse_enabled()
def send_environment_update_message_for_environment(environment):
    if False:
        while True:
            i = 10
    tasks.send_environment_update_message.delay(args=(environment.api_key, environment.updated_at.isoformat()))