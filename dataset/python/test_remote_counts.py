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
        self.server_uuid = "6cde5f7a-1f7e-4978-9716-49f69ebfc9fe"
        self.server = RemoteZulipServer(
            uuid=self.server_uuid,
            api_key="magic_secret_api_key",
            hostname="demo.example.com",
            last_updated=timezone_now(),
        )
        self.server.save()
        super().setUp()

    def test_compute_max_monthly_messages(self) -> None:
        now = timezone_now()
        # Note: We will use this modified now_offset value to subtract N days from it,
        # to simulate the data in the time series for the day now - N days. This avoids
        # inconsistent behavior on the boundaries. E.g. does an entry with
        # end_time=now - 30 days belong to the "last 30 days" interval or the 30 days before that?
        # Using now_offset avoids this ambiguity.
        now_offset = now + datetime.timedelta(hours=1)

        # First try with absolutely no analytics data.
        with self.assertRaises(MissingDataError):
            compute_max_monthly_messages(self.server)

        # This one-off row is just because we use this property as a proxy for
        # "the server submitted useful analytics data" in compute_max_monthly_messages.
        # Servers without such an entry raises MissingDataError as illustrated above.
        # See the function's implementation for details.
        RemoteInstallationCount.objects.create(
            server=self.server,
            remote_id=1,
            property="active_users_audit:is_bot:day",
            value=5,
            end_time=now_offset - datetime.timedelta(days=4),
        )

        # If we're missing any message data (which is the same as message data with 0, because
        # we actually don't record 0s), then the function should just very reasonably return 0.
        self.assertEqual(compute_max_monthly_messages(self.server), 0)

        # Last 30 days of data:
        RemoteInstallationCount.objects.bulk_create(
            RemoteInstallationCount(
                server=self.server,
                remote_id=1,
                property="messages_sent:message_type:day",
                value=10,
                end_time=now_offset - datetime.timedelta(days=t),
            )
            for t in range(1, 31)
        )
        # 30 days before that:
        # This will be the peak of the last 3 months - with 900 messages total
        RemoteInstallationCount.objects.bulk_create(
            RemoteInstallationCount(
                server=self.server,
                remote_id=1,
                property="messages_sent:message_type:day",
                value=30,
                end_time=now_offset - datetime.timedelta(days=30 + t),
            )
            for t in range(1, 31)
        )
        # Additional 30 days before that:
        # This is the last month we're considering for the calculation
        RemoteInstallationCount.objects.bulk_create(
            RemoteInstallationCount(
                server=self.server,
                remote_id=1,
                property="messages_sent:message_type:day",
                value=20,
                end_time=now_offset - datetime.timedelta(days=60 + t),
            )
            for t in range(1, 31)
        )
        # Additional 30 days before that:
        # This is >90 days ago and should be ignored for the calculation. We simulate the highest
        # amounts of messages here, to test that this is indeed ignored.
        RemoteInstallationCount.objects.bulk_create(
            RemoteInstallationCount(
                server=self.server,
                remote_id=1,
                property="messages_sent:message_type:day",
                value=100,
                end_time=now_offset - datetime.timedelta(days=90 + t),
            )
            for t in range(1, 31)
        )

        with time_machine.travel(now, tick=False):
            self.assertEqual(compute_max_monthly_messages(self.server), 900)
