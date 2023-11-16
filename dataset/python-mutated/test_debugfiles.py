from sentry.models.debugfile import ProjectDebugFile
from sentry.models.files.file import File
from sentry.tasks.deletion.scheduled import run_scheduled_deletions
from sentry.testutils.cases import TransactionTestCase
from sentry.testutils.hybrid_cloud import HybridCloudTestMixin
from sentry.testutils.silo import region_silo_test

@region_silo_test(stable=True)
class DeleteDebugFilesTest(TransactionTestCase, HybridCloudTestMixin):

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        dif = self.create_dif_file()
        dif2 = self.create_dif_file()
        self.ScheduledDeletion.schedule(instance=dif, days=0)
        with self.tasks():
            run_scheduled_deletions()
        assert not ProjectDebugFile.objects.filter(id=dif.id).exists()
        assert not File.objects.filter(id=dif.file.id).exists()
        assert ProjectDebugFile.objects.filter(id=dif2.id).exists()
        assert File.objects.filter(id=dif2.file.id).exists()