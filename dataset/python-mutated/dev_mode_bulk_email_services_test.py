"""Tests for dev mode bulk email services."""
from __future__ import annotations
import logging
from core.platform.bulk_email import dev_mode_bulk_email_services
from core.tests import test_utils

class DevModeBulkEmailServicesUnitTests(test_utils.GenericTestBase):
    """Tests for mailchimp services."""

    def test_add_or_update_user_status(self) -> None:
        if False:
            while True:
                i = 10
        observed_log_messages = []

        def _mock_logging_function(msg: str, *args: str) -> None:
            if False:
                print('Hello World!')
            'Mocks logging.info().'
            observed_log_messages.append(msg % args)
        with self.swap(logging, 'info', _mock_logging_function):
            dev_mode_bulk_email_services.add_or_update_user_status('test@example.com', {}, 'Web', can_receive_email_updates=True)
            self.assertItemsEqual(observed_log_messages, ["Updated status of email ID test@example.com's bulk email preference in the service provider's db to True. Cannot access API, since this is a dev environment."])
            observed_log_messages = []
            dev_mode_bulk_email_services.add_or_update_user_status('test@example.com', {}, 'Web', can_receive_email_updates=False)
            self.assertItemsEqual(observed_log_messages, ["Updated status of email ID test@example.com's bulk email preference in the service provider's db to False. Cannot access API, since this is a dev environment."])

    def test_permanently_delete_user(self) -> None:
        if False:
            print('Hello World!')
        observed_log_messages = []

        def _mock_logging_function(msg: str, *args: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            'Mocks logging.info().'
            observed_log_messages.append(msg % args)
        with self.swap(logging, 'info', _mock_logging_function):
            dev_mode_bulk_email_services.permanently_delete_user_from_list('test@example.com')
            self.assertItemsEqual(observed_log_messages, ["Email ID test@example.com permanently deleted from bulk email provider's db. Cannot access API, since this is a dev environment"])