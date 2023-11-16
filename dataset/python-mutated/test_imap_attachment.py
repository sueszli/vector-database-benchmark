from __future__ import annotations
from unittest.mock import Mock, patch
import pytest
from airflow.providers.imap.sensors.imap_attachment import ImapAttachmentSensor

class TestImapAttachmentSensor:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.kwargs = dict(attachment_name='test_file', check_regex=False, mail_folder='INBOX', mail_filter='All', task_id='test_task', dag=None)

    @pytest.mark.parametrize('has_attachment_return_value', [True, False])
    @patch('airflow.providers.imap.sensors.imap_attachment.ImapHook')
    def test_poke(self, mock_imap_hook, has_attachment_return_value):
        if False:
            i = 10
            return i + 15
        mock_imap_hook.return_value.__enter__ = Mock(return_value=mock_imap_hook)
        mock_imap_hook.has_mail_attachment.return_value = has_attachment_return_value
        has_attachment = ImapAttachmentSensor(**self.kwargs).poke(context={})
        assert has_attachment == mock_imap_hook.has_mail_attachment.return_value
        mock_imap_hook.has_mail_attachment.assert_called_once_with(name=self.kwargs['attachment_name'], check_regex=self.kwargs['check_regex'], mail_folder=self.kwargs['mail_folder'], mail_filter=self.kwargs['mail_filter'])