import logging
from celery import subtask
from celery.signals import worker_ready, worker_shutdown, after_setup_logger
from django.core.cache import cache
from django_celery_beat.models import PeriodicTask
from common.utils import get_logger
from .decorator import get_after_app_ready_tasks, get_after_app_shutdown_clean_tasks
from .logger import CeleryThreadTaskFileHandler
logger = get_logger(__file__)
safe_str = lambda x: x

@worker_ready.connect
def on_app_ready(sender=None, headers=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    if cache.get('CELERY_APP_READY', 0) == 1:
        return
    cache.set('CELERY_APP_READY', 1, 10)
    tasks = get_after_app_ready_tasks()
    logger.debug('Work ready signal recv')
    logger.debug('Start need start task: [{}]'.format(', '.join(tasks)))
    for task in tasks:
        subtask(task).delay()

@worker_shutdown.connect
def after_app_shutdown_periodic_tasks(sender=None, **kwargs):
    if False:
        return 10
    if cache.get('CELERY_APP_SHUTDOWN', 0) == 1:
        return
    cache.set('CELERY_APP_SHUTDOWN', 1, 10)
    tasks = get_after_app_shutdown_clean_tasks()
    logger.debug('Worker shutdown signal recv')
    logger.debug('Clean period tasks: [{}]'.format(', '.join(tasks)))
    PeriodicTask.objects.filter(name__in=tasks).delete()

@after_setup_logger.connect
def add_celery_logger_handler(sender=None, logger=None, loglevel=None, format=None, **kwargs):
    if False:
        while True:
            i = 10
    if not logger:
        return
    task_handler = CeleryThreadTaskFileHandler()
    task_handler.setLevel(loglevel)
    formatter = logging.Formatter(format)
    task_handler.setFormatter(formatter)
    logger.addHandler(task_handler)