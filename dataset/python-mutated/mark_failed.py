from __future__ import annotations
import logging
import uuid
from datetime import datetime
from django.db.models import Q
from django.utils import timezone
from sentry import features
from sentry.grouping.utils import hash_from_values
from sentry.issues.grouptype import MonitorCheckInFailure, MonitorCheckInMissed, MonitorCheckInTimeout
from sentry.issues.producer import PayloadType
from sentry.models.organization import Organization
from sentry.monitors.constants import SUBTITLE_DATETIME_FORMAT, TIMEOUT
from sentry.monitors.models import CheckInStatus, MonitorCheckIn, MonitorEnvironment, MonitorIncident, MonitorObjectStatus, MonitorStatus
logger = logging.getLogger(__name__)

def mark_failed(failed_checkin: MonitorCheckIn, ts: datetime):
    if False:
        return 10
    "\n    Given a failing check-in, mark the monitor environment as failed and trigger\n    side effects for creating monitor incidents and issues.\n\n    The provided `ts` is the reference time for when the next check-in time is\n    calculated from. This typically would be the failed check-in's `date_added`\n    or completion time. Though for the missed and timedout tasks this may be\n    computed based on the tasks reference time.\n    "
    monitor_env = failed_checkin.monitor_environment
    failure_issue_threshold = monitor_env.monitor.config.get('failure_issue_threshold', 0)
    next_checkin = monitor_env.monitor.get_next_expected_checkin(ts)
    next_checkin_latest = monitor_env.monitor.get_next_expected_checkin_latest(ts)
    if failed_checkin.status == CheckInStatus.MISSED:
        last_checkin = monitor_env.last_checkin
    else:
        last_checkin = failed_checkin.date_added
    monitors_to_update = MonitorEnvironment.objects.filter(Q(last_checkin__lte=last_checkin) | Q(last_checkin__isnull=True), id=monitor_env.id)
    field_updates = {'last_checkin': last_checkin, 'next_checkin': next_checkin, 'next_checkin_latest': next_checkin_latest}
    if not failure_issue_threshold:
        failed_status_map = {CheckInStatus.MISSED: MonitorStatus.MISSED_CHECKIN, CheckInStatus.TIMEOUT: MonitorStatus.TIMEOUT}
        field_updates['status'] = failed_status_map.get(failed_checkin.status, MonitorStatus.ERROR)
    affected = monitors_to_update.update(**field_updates)
    if not affected:
        return False
    monitor_env.refresh_from_db()
    if failure_issue_threshold:
        return mark_failed_threshold(failed_checkin, failure_issue_threshold)
    else:
        return mark_failed_no_threshold(failed_checkin)

def mark_failed_threshold(failed_checkin: MonitorCheckIn, failure_issue_threshold: int):
    if False:
        return 10
    from sentry.signals import monitor_environment_failed
    monitor_env = failed_checkin.monitor_environment
    monitor_disabled = monitor_env.monitor.status == MonitorObjectStatus.DISABLED
    fingerprint = None
    if monitor_env.status == MonitorStatus.OK:
        previous_checkins = list(reversed(MonitorCheckIn.objects.filter(monitor_environment=monitor_env).order_by('-date_added').values('id', 'date_added', 'status')[:failure_issue_threshold]))
        if not all([checkin['status'] not in [CheckInStatus.IN_PROGRESS, CheckInStatus.OK] for checkin in previous_checkins]):
            return False
        monitor_env.status = MonitorStatus.ERROR
        monitor_env.last_state_change = monitor_env.last_checkin
        monitor_env.save(update_fields=('status', 'last_state_change'))
        if not monitor_disabled:
            starting_checkin = previous_checkins[0]
            fingerprint = hash_from_values([uuid.uuid4()])
            MonitorIncident.objects.create(monitor=monitor_env.monitor, monitor_environment=monitor_env, starting_checkin_id=starting_checkin['id'], starting_timestamp=starting_checkin['date_added'], grouphash=fingerprint)
    elif monitor_env.status in [MonitorStatus.ERROR, MonitorStatus.MISSED_CHECKIN, MonitorStatus.TIMEOUT]:
        previous_checkins = [MonitorCheckIn.objects.filter(monitor_environment=monitor_env).order_by('-date_added').values('id', 'date_added', 'status').first()]
        fingerprint = monitor_env.incident_grouphash
    else:
        return False
    if monitor_disabled:
        return True
    for previous_checkin in previous_checkins:
        checkin_from_db = MonitorCheckIn.objects.get(id=previous_checkin['id'])
        create_issue_platform_occurrence(checkin_from_db, fingerprint)
    monitor_environment_failed.send(monitor_environment=monitor_env, sender=type(monitor_env))
    return True

def mark_failed_no_threshold(failed_checkin: MonitorCheckIn):
    if False:
        i = 10
        return i + 15
    from sentry.signals import monitor_environment_failed
    monitor_env = failed_checkin.monitor_environment
    if monitor_env.monitor.status == MonitorObjectStatus.DISABLED:
        return True
    use_issue_platform = False
    try:
        organization = Organization.objects.get(id=monitor_env.monitor.organization_id)
        use_issue_platform = features.has('organizations:issue-platform', organization=organization)
    except Organization.DoesNotExist:
        pass
    if use_issue_platform:
        create_issue_platform_occurrence(failed_checkin)
    else:
        create_legacy_event(failed_checkin)
    monitor_environment_failed.send(monitor_environment=monitor_env, sender=type(monitor_env))
    return True

def create_legacy_event(failed_checkin: MonitorCheckIn):
    if False:
        i = 10
        return i + 15
    from sentry.coreapi import insert_data_to_database_legacy
    from sentry.event_manager import EventManager
    from sentry.models.project import Project
    monitor_env = failed_checkin.monitor_environment
    context = get_monitor_environment_context(monitor_env)
    reason_map = {CheckInStatus.MISSED: 'missed_checkin', CheckInStatus.TIMEOUT: 'duration'}
    reason = reason_map.get(failed_checkin.status, 'unknown')
    event_manager = EventManager({'logentry': {'message': f'Monitor failure: {monitor_env.monitor.name} ({reason})'}, 'contexts': {'monitor': context}, 'fingerprint': ['monitor', str(monitor_env.monitor.guid), reason], 'environment': monitor_env.environment.name, 'tags': {'monitor.id': str(monitor_env.monitor.guid), 'monitor.slug': monitor_env.monitor.slug}}, project=Project(id=monitor_env.monitor.project_id))
    event_manager.normalize()
    data = event_manager.get_data()
    insert_data_to_database_legacy(data)

def create_issue_platform_occurrence(failed_checkin: MonitorCheckIn, fingerprint=None):
    if False:
        print('Hello World!')
    from sentry.issues.issue_occurrence import IssueEvidence, IssueOccurrence
    from sentry.issues.producer import produce_occurrence_to_kafka
    monitor_env = failed_checkin.monitor_environment
    current_timestamp = datetime.utcnow().replace(tzinfo=timezone.utc)
    occurrence_data = get_occurrence_data(failed_checkin)
    last_successful_checkin_timestamp = 'None'
    last_successful_checkin = monitor_env.get_last_successful_checkin()
    if last_successful_checkin:
        last_successful_checkin_timestamp = last_successful_checkin.date_added.isoformat()
    occurrence = IssueOccurrence(id=uuid.uuid4().hex, resource_id=None, project_id=monitor_env.monitor.project_id, event_id=uuid.uuid4().hex, fingerprint=[fingerprint if fingerprint else hash_from_values(['monitor', str(monitor_env.monitor.guid), occurrence_data['reason']])], type=occurrence_data['group_type'], issue_title=f'Monitor failure: {monitor_env.monitor.name}', subtitle=occurrence_data['subtitle'], evidence_display=[IssueEvidence(name='Failure reason', value=occurrence_data['reason'], important=True), IssueEvidence(name='Environment', value=monitor_env.environment.name, important=False), IssueEvidence(name='Last successful check-in', value=last_successful_checkin_timestamp, important=False)], evidence_data={}, culprit=occurrence_data['reason'], detection_time=current_timestamp, level=occurrence_data['level'])
    if failed_checkin.trace_id:
        trace_id = failed_checkin.trace_id.hex
    else:
        trace_id = None
    event_data = {'contexts': {'monitor': get_monitor_environment_context(monitor_env)}, 'environment': monitor_env.environment.name, 'event_id': occurrence.event_id, 'fingerprint': fingerprint if fingerprint else ['monitor', str(monitor_env.monitor.guid), occurrence_data['reason']], 'platform': 'other', 'project_id': monitor_env.monitor.project_id, 'received': current_timestamp.isoformat(), 'sdk': None, 'tags': {'monitor.id': str(monitor_env.monitor.guid), 'monitor.slug': str(monitor_env.monitor.slug)}, 'timestamp': current_timestamp.isoformat()}
    if trace_id:
        event_data['contexts']['trace'] = {'trace_id': trace_id, 'span_id': None}
    produce_occurrence_to_kafka(payload_type=PayloadType.OCCURRENCE, occurrence=occurrence, event_data=event_data)

def get_monitor_environment_context(monitor_environment: MonitorEnvironment):
    if False:
        i = 10
        return i + 15
    config = monitor_environment.monitor.config.copy()
    if 'schedule_type' in config:
        config['schedule_type'] = monitor_environment.monitor.get_schedule_type_display()
    return {'id': str(monitor_environment.monitor.guid), 'slug': str(monitor_environment.monitor.slug), 'name': monitor_environment.monitor.name, 'config': monitor_environment.monitor.config, 'status': monitor_environment.get_status_display(), 'type': monitor_environment.monitor.get_type_display()}

def get_occurrence_data(checkin: MonitorCheckIn):
    if False:
        for i in range(10):
            print('nop')
    if checkin.status == CheckInStatus.MISSED:
        expected_time = checkin.expected_time.astimezone(checkin.monitor.timezone).strftime(SUBTITLE_DATETIME_FORMAT) if checkin.expected_time else 'the expected time'
        return {'group_type': MonitorCheckInMissed, 'level': 'warning', 'reason': 'missed_checkin', 'subtitle': f'No check-in reported on {expected_time}.'}
    if checkin.status == CheckInStatus.TIMEOUT:
        duration = (checkin.monitor.config or {}).get('max_runtime') or TIMEOUT
        return {'group_type': MonitorCheckInTimeout, 'level': 'error', 'reason': 'duration', 'subtitle': f'Check-in exceeded maximum duration of {duration} minutes.'}
    return {'group_type': MonitorCheckInFailure, 'level': 'error', 'reason': 'error', 'subtitle': 'An error occurred during the latest check-in.'}