from datetime import datetime
from sentry.monitors.models import CheckInStatus, MonitorCheckIn, MonitorEnvironment, MonitorIncident, MonitorObjectStatus, MonitorStatus

def mark_ok(checkin: MonitorCheckIn, ts: datetime):
    if False:
        for i in range(10):
            print('nop')
    monitor_env = checkin.monitor_environment
    next_checkin = monitor_env.monitor.get_next_expected_checkin(ts)
    next_checkin_latest = monitor_env.monitor.get_next_expected_checkin_latest(ts)
    params = {'last_checkin': checkin.date_added, 'next_checkin': next_checkin, 'next_checkin_latest': next_checkin_latest}
    if monitor_env.monitor.status != MonitorObjectStatus.DISABLED and monitor_env.status != MonitorStatus.OK:
        params['status'] = MonitorStatus.OK
        recovery_threshold = monitor_env.monitor.config.get('recovery_threshold')
        if recovery_threshold:
            previous_checkins = MonitorCheckIn.objects.filter(monitor_environment=monitor_env).values('id', 'date_added', 'status').order_by('-date_added')[:recovery_threshold]
            incident_recovering = all((previous_checkin['status'] == CheckInStatus.OK for previous_checkin in previous_checkins))
            if incident_recovering:
                MonitorIncident.objects.filter(monitor_environment=monitor_env, grouphash=monitor_env.incident_grouphash).update(resolving_checkin=checkin, resolving_timestamp=checkin.date_added)
                params['last_state_change'] = ts
            else:
                params.pop('status', None)
    MonitorEnvironment.objects.filter(id=monitor_env.id).exclude(last_checkin__gt=ts).update(**params)