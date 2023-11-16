from datetime import timedelta
from uuid import uuid4
import pytest
from django.utils import timezone
from sentry.models.outbox import outbox_context
from sentry.testutils.cases import TestMigrations

@pytest.mark.skip('Migration 581 makes alert rule selects fail here.')
class MonitorsFixNextCheckinLatestMigrationTest(TestMigrations):
    migrate_from = '0567_add_slug_reservation_model'
    migrate_to = '0568_monitors_fix_next_checkin_latest'

    def setup_before_migration(self, apps):
        if False:
            while True:
                i = 10
        with outbox_context(flush=False):
            self.now = timezone.now().replace(second=0, microsecond=0)
            Monitor = apps.get_model('sentry', 'Monitor')
            MonitorEnvironment = apps.get_model('sentry', 'MonitorEnvironment')
            self.monitor1 = Monitor.objects.create(guid=uuid4(), slug='test1', organization_id=self.organization.id, project_id=self.project.id, config={'schedule': '* * * * *', 'checkin_margin': None, 'max_runtime': None})
            self.monitor_env1 = MonitorEnvironment.objects.create(monitor=self.monitor1, environment_id=self.environment.id, next_checkin=self.now, next_checkin_latest=self.now)
            self.monitor2 = Monitor.objects.create(guid=uuid4(), slug='test2', organization_id=self.organization.id, project_id=self.project.id, config={'schedule': '* * * * *', 'checkin_margin': 3, 'max_runtime': None})
            self.monitor_env2 = MonitorEnvironment.objects.create(monitor=self.monitor2, environment_id=self.environment.id, next_checkin=self.now, next_checkin_latest=self.now)
            self.monitor3 = Monitor.objects.create(guid=uuid4(), slug='test3', organization_id=self.organization.id, project_id=self.project.id, config={'schedule': '* * * * *', 'checkin_margin': 3, 'max_runtime': None})
            self.monitor_env3 = MonitorEnvironment.objects.create(monitor=self.monitor3, environment_id=self.environment.id, next_checkin=self.now, next_checkin_latest=self.now + timedelta(minutes=5))

    def test(self):
        if False:
            return 10
        self.monitor_env1.refresh_from_db()
        self.monitor_env2.refresh_from_db()
        self.monitor_env3.refresh_from_db()
        assert self.monitor_env1.next_checkin_latest == self.now + timedelta(minutes=1)
        assert self.monitor_env2.next_checkin_latest == self.now + timedelta(minutes=3)
        assert self.monitor_env3.next_checkin_latest == self.now + timedelta(minutes=5)