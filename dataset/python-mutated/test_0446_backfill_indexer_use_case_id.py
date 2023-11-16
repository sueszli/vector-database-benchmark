import pytest
from sentry.sentry_metrics.indexer.postgres.models import PerfStringIndexer
from sentry.testutils.cases import TestMigrations

@pytest.mark.skip('Migration is no longer runnable. Retain until migration is removed.')
class PerfIndexerUseCaseIdBackfillTest(TestMigrations):
    migrate_from = '0445_drop_deprecated_monitor_next_last_checkin_db_op'
    migrate_to = '0446_backfill_indexer_use_case_id'

    def setup_before_migration(self, apps):
        if False:
            for i in range(10):
                print('nop')
        PerfStringIndexer = apps.get_model('sentry', 'PerfStringIndexer')
        PerfStringIndexer.objects.create(string='hello', organization_id=12, use_case_id='performance')
        PerfStringIndexer.objects.create(string='bye', organization_id=12, use_case_id='performance')
        PerfStringIndexer.objects.create(string='bye', organization_id=12, use_case_id='transactions')

    def test(self):
        if False:
            while True:
                i = 10
        assert not PerfStringIndexer.objects.filter(string='hello', use_case_id='performance')
        assert len(PerfStringIndexer.objects.filter(string='bye')) == 2