import os
import time
from datetime import timedelta
from typing import Any, Dict
from django.core.management.base import BaseCommand
from django.utils.timezone import now as timezone_now
from typing_extensions import override
from analytics.lib.counts import ALL_COUNT_STATS, CountStat
from analytics.models import installation_epoch
from zerver.lib.timestamp import TimeZoneNotUTCError, floor_to_day, floor_to_hour, verify_UTC
from zerver.models import Realm
states = {0: 'OK', 1: 'WARNING', 2: 'CRITICAL', 3: 'UNKNOWN'}

class Command(BaseCommand):
    help = 'Checks FillState table.\n\n    Run as a cron job that runs every hour.'

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        fill_state = self.get_fill_state()
        status = fill_state['status']
        message = fill_state['message']
        state_file_path = '/var/lib/nagios_state/check-analytics-state'
        state_file_tmp = state_file_path + '-tmp'
        with open(state_file_tmp, 'w') as f:
            f.write(f'{int(time.time())}|{status}|{states[status]}|{message}\n')
        os.rename(state_file_tmp, state_file_path)

    def get_fill_state(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        if not Realm.objects.exists():
            return {'status': 0, 'message': 'No realms exist, so not checking FillState.'}
        warning_unfilled_properties = []
        critical_unfilled_properties = []
        for (property, stat) in ALL_COUNT_STATS.items():
            last_fill = stat.last_successful_fill()
            if last_fill is None:
                last_fill = installation_epoch()
            try:
                verify_UTC(last_fill)
            except TimeZoneNotUTCError:
                return {'status': 2, 'message': f'FillState not in UTC for {property}'}
            if stat.frequency == CountStat.DAY:
                floor_function = floor_to_day
                warning_threshold = timedelta(hours=26)
                critical_threshold = timedelta(hours=50)
            else:
                floor_function = floor_to_hour
                warning_threshold = timedelta(minutes=90)
                critical_threshold = timedelta(minutes=150)
            if floor_function(last_fill) != last_fill:
                return {'status': 2, 'message': f'FillState not on {stat.frequency} boundary for {property}'}
            time_to_last_fill = timezone_now() - last_fill
            if time_to_last_fill > critical_threshold:
                critical_unfilled_properties.append(property)
            elif time_to_last_fill > warning_threshold:
                warning_unfilled_properties.append(property)
        if len(critical_unfilled_properties) == 0 and len(warning_unfilled_properties) == 0:
            return {'status': 0, 'message': 'FillState looks fine.'}
        if len(critical_unfilled_properties) == 0:
            return {'status': 1, 'message': 'Missed filling {} once.'.format(', '.join(warning_unfilled_properties))}
        return {'status': 2, 'message': 'Missed filling {} once. Missed filling {} at least twice.'.format(', '.join(warning_unfilled_properties), ', '.join(critical_unfilled_properties))}