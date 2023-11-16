from __future__ import annotations
from unittest import mock
from airflow.providers.amazon.aws.transfers.imap_attachment_to_s3 import ImapAttachmentToS3Operator

class TestImapAttachmentToS3Operator:

    def setup_method(self):
        if False:
            print('Hello World!')
        self.kwargs = dict(imap_attachment_name='test_file', s3_bucket='test_bucket', s3_key='test_file', imap_check_regex=False, imap_mail_folder='INBOX', imap_mail_filter='All', s3_overwrite=False, task_id='test_task', dag=None)

    @mock.patch('airflow.providers.amazon.aws.transfers.imap_attachment_to_s3.S3Hook')
    @mock.patch('airflow.providers.amazon.aws.transfers.imap_attachment_to_s3.ImapHook')
    def test_execute(self, mock_imap_hook, mock_s3_hook):
        if False:
            i = 10
            return i + 15
        mock_imap_hook.return_value.__enter__ = mock_imap_hook
        mock_imap_hook.return_value.retrieve_mail_attachments.return_value = [('test_file', b'Hello World')]
        ImapAttachmentToS3Operator(**self.kwargs).execute(context={})
        mock_imap_hook.return_value.retrieve_mail_attachments.assert_called_once_with(name=self.kwargs['imap_attachment_name'], check_regex=self.kwargs['imap_check_regex'], latest_only=True, mail_folder=self.kwargs['imap_mail_folder'], mail_filter=self.kwargs['imap_mail_filter'])
        mock_s3_hook.return_value.load_bytes.assert_called_once_with(bytes_data=mock_imap_hook.return_value.retrieve_mail_attachments.return_value[0][1], bucket_name=self.kwargs['s3_bucket'], key=self.kwargs['s3_key'], replace=self.kwargs['s3_overwrite'])