import logging
import os
from celery import Celery
from celery.signals import setup_logging
from django.conf import settings
from .plugins import discover_plugins_modules
CELERY_LOGGER_NAME = 'celery'

@setup_logging.connect
def setup_celery_logging(loglevel=None, **kwargs):
    if False:
        print('Hello World!')
    'Skip default Celery logging configuration.\n\n    Will rely on Django to set up the base root logger.\n    Celery loglevel will be set if provided as Celery command argument.\n    '
    if loglevel:
        logging.getLogger(CELERY_LOGGER_NAME).setLevel(loglevel)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'saleor.settings')
app = Celery('saleor')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
app.autodiscover_tasks(lambda : discover_plugins_modules(settings.PLUGINS))
app.autodiscover_tasks(related_name='search_tasks')