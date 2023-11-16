import logging
import typing
logger = logging.getLogger(__name__)
registered_tasks: typing.Dict[str, typing.Callable] = {}

def register_task(task_identifier: str, callable_: typing.Callable):
    if False:
        return 10
    global registered_tasks
    logger.debug("Registering task '%s'", task_identifier)
    registered_tasks[task_identifier] = callable_
    logger.debug('Registered tasks now has the following tasks registered: %s', list(registered_tasks.keys()))

def get_task(task_identifier: str) -> typing.Callable:
    if False:
        i = 10
        return i + 15
    return registered_tasks[task_identifier]