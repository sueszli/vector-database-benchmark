import logging
from awx.main.analytics.subsystem_metrics import Metrics
from awx.main.dispatch.publish import task
from awx.main.dispatch import get_task_queuename
logger = logging.getLogger('awx.main.scheduler')

@task(queue=get_task_queuename)
def send_subsystem_metrics():
    if False:
        while True:
            i = 10
    Metrics().send_metrics()