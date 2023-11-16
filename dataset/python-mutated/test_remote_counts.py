import datetime
import time_machine
from django.utils.timezone import now as timezone_now
from typing_extensions import override
from zerver.lib.test_classes import ZulipTestCase
from zilencer.lib.remote_counts import MissingDataError, compute_max_monthly_messages
from zilencer.models import RemoteInstallationCount, RemoteZulipServer

class RemoteCountTest(ZulipTestCase):

    @override
    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.server_uuid = '6cde5f7a-1f7e-4978-9716-49f69ebfc9fe'
        self.server = RemoteZulipServer(uuid=self.server_uuid, api_key='magic_secret_api_key', hostname='demo.example.com', last_updated=timezone_now())
        self.server.save()
        super().setUp()

    def test_compute_max_monthly_messages(self) -> None:
        if False:
            i = 10
            return i + 15
        now = timezone_now()
        now_offset = now + datetime.timedelta(hours=1)
        with self.assertRaises(MissingDataError):
            compute_max_monthly_messages(self.server)
        RemoteInstallationCount.objects.create(server=self.server, remote_id=1, property='active_users_audit:is_bot:day', value=5, end_time=now_offset - datetime.timedelta(days=4))
        self.assertEqual(compute_max_monthly_messages(self.server), 0)
        RemoteInstallationCount.objects.bulk_create((RemoteInstallationCount(server=self.server, remote_id=1, property='messages_sent:message_type:day', value=10, end_time=now_offset - datetime.timedelta(days=t)) for t in range(1, 31)))
        RemoteInstallationCount.objects.bulk_create((RemoteInstallationCount(server=self.server, remote_id=1, property='messages_sent:message_type:day', value=30, end_time=now_offset - datetime.timedelta(days=30 + t)) for t in range(1, 31)))
        RemoteInstallationCount.objects.bulk_create((RemoteInstallationCount(server=self.server, remote_id=1, property='messages_sent:message_type:day', value=20, end_time=now_offset - datetime.timedelta(days=60 + t)) for t in range(1, 31)))
        RemoteInstallationCount.objects.bulk_create((RemoteInstallationCount(server=self.server, remote_id=1, property='messages_sent:message_type:day', value=100, end_time=now_offset - datetime.timedelta(days=90 + t)) for t in range(1, 31)))
        with time_machine.travel(now, tick=False):
            self.assertEqual(compute_max_monthly_messages(self.server), 900)