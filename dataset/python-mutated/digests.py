import logging
import time
from typing import List, Optional
from sentry.digests import Record, get_option_key
from sentry.digests.backends.base import InvalidState
from sentry.digests.notifications import build_digest, split_key
from sentry.models.options.project_option import ProjectOption
from sentry.models.project import Project
from sentry.silo import SiloMode
from sentry.tasks.base import instrumented_task
from sentry.utils import snuba
logger = logging.getLogger(__name__)

@instrumented_task(name='sentry.tasks.digests.schedule_digests', queue='digests.scheduling', silo_mode=SiloMode.REGION)
def schedule_digests():
    if False:
        print('Hello World!')
    from sentry import digests
    deadline = time.time()
    timeout = 300
    digests.maintenance(deadline - timeout)
    for entry in digests.schedule(deadline):
        deliver_digest.delay(entry.key, entry.timestamp)

@instrumented_task(name='sentry.tasks.digests.deliver_digest', queue='digests.delivery', silo_mode=SiloMode.REGION)
def deliver_digest(key, schedule_timestamp=None, notification_uuid: Optional[str]=None):
    if False:
        for i in range(10):
            print('nop')
    from sentry import digests
    from sentry.mail import mail_adapter
    try:
        (project, target_type, target_identifier, fallthrough_choice) = split_key(key)
    except Project.DoesNotExist as error:
        logger.info(f'Cannot deliver digest {key} due to error: {error}')
        digests.delete(key)
        return
    minimum_delay = ProjectOption.objects.get_value(project, get_option_key('mail', 'minimum_delay'))
    with snuba.options_override({'consistent': True}):
        try:
            with digests.digest(key, minimum_delay=minimum_delay) as records:
                (digest, logs) = build_digest(project, records)
                if not notification_uuid:
                    notification_uuid = get_notification_uuid_from_records(records)
        except InvalidState as error:
            logger.info(f'Skipped digest delivery: {error}', exc_info=True)
            return
        if digest:
            mail_adapter.notify_digest(project, digest, target_type, target_identifier, fallthrough_choice=fallthrough_choice, notification_uuid=notification_uuid)
        else:
            logger.info('Skipped digest delivery due to empty digest', extra={'project': project.id, 'target_type': target_type.value, 'target_identifier': target_identifier, 'build_digest_logs': logs, 'fallthrough_choice': fallthrough_choice.value if fallthrough_choice else None})

def get_notification_uuid_from_records(records: List[Record]) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    for record in records:
        try:
            notification_uuid = record.value.notification_uuid
            if notification_uuid:
                return notification_uuid
        except Exception:
            return None
    return None