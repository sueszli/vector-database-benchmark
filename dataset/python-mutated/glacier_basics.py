"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Simple Storage Service
Glacier to create and manage vaults and archives.
"""
import argparse
import logging
import os
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class GlacierWrapper:
    """Encapsulates Amazon S3 Glacier API operations."""

    def __init__(self, glacier_resource):
        if False:
            i = 10
            return i + 15
        '\n        :param glacier_resource: A Boto3 Amazon S3 Glacier resource.\n        '
        self.glacier_resource = glacier_resource

    def create_vault(self, vault_name):
        if False:
            print('Hello World!')
        '\n        Creates a vault.\n\n        :param vault_name: The name to give the vault.\n        :return: The newly created vault.\n        '
        try:
            vault = self.glacier_resource.create_vault(vaultName=vault_name)
            logger.info('Created vault %s.', vault_name)
        except ClientError:
            logger.exception("Couldn't create vault %s.", vault_name)
            raise
        else:
            return vault

    def list_vaults(self):
        if False:
            while True:
                i = 10
        '\n        Lists vaults for the current account.\n        '
        try:
            for vault in self.glacier_resource.vaults.all():
                logger.info('Got vault %s.', vault.name)
        except ClientError:
            logger.exception("Couldn't list vaults.")
            raise

    @staticmethod
    def upload_archive(vault, archive_description, archive_file):
        if False:
            print('Hello World!')
        '\n        Uploads an archive to a vault.\n\n        :param vault: The vault where the archive is put.\n        :param archive_description: A description of the archive.\n        :param archive_file: The archive file to put in the vault.\n        :return: The uploaded archive.\n        '
        try:
            archive = vault.upload_archive(archiveDescription=archive_description, body=archive_file)
            logger.info('Uploaded %s with ID %s to vault %s.', archive_description, archive.id, vault.name)
        except ClientError:
            logger.exception("Couldn't upload %s to %s.", archive_description, vault.name)
            raise
        else:
            return archive

    @staticmethod
    def initiate_inventory_retrieval(vault):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initiates an inventory retrieval job. The inventory describes the contents\n        of the vault. Standard retrievals typically complete within 3—5 hours.\n        When the job completes, you can get the inventory by calling get_output().\n\n        :param vault: The vault to inventory.\n        :return: The inventory retrieval job.\n        '
        try:
            job = vault.initiate_inventory_retrieval()
            logger.info('Started %s job with ID %s.', job.action, job.id)
        except ClientError:
            logger.exception("Couldn't start job on vault %s.", vault.name)
            raise
        else:
            return job

    @staticmethod
    def list_jobs(vault, job_type):
        if False:
            while True:
                i = 10
        '\n        Lists jobs by type for the specified vault.\n\n        :param vault: The vault to query.\n        :param job_type: The type of job to list.\n        :return: The list of jobs of the requested type.\n        '
        job_list = []
        try:
            if job_type == 'all':
                jobs = vault.jobs.all()
            elif job_type == 'in_progress':
                jobs = vault.jobs_in_progress.all()
            elif job_type == 'completed':
                jobs = vault.completed_jobs.all()
            elif job_type == 'succeeded':
                jobs = vault.succeeded_jobs.all()
            elif job_type == 'failed':
                jobs = vault.failed_jobs.all()
            else:
                jobs = []
                logger.warning("%s isn't a type of job I can get.", job_type)
            for job in jobs:
                job_list.append(job)
                logger.info('Got %s %s job %s.', job_type, job.action, job.id)
        except ClientError:
            logger.exception("Couldn't get %s jobs from %s.", job_type, vault.name)
            raise
        else:
            return job_list

    @staticmethod
    def delete_vault(vault):
        if False:
            return 10
        '\n        Deletes a vault.\n\n        :param vault: The vault to delete.\n        '
        try:
            vault.delete()
            logger.info('Deleted vault %s.', vault.name)
        except ClientError:
            logger.exception("Couldn't delete vault %s.", vault.name)
            raise

    @staticmethod
    def initiate_archive_retrieval(archive):
        if False:
            while True:
                i = 10
        '\n        Initiates an archive retrieval job. Standard retrievals typically complete\n        within 3—5 hours. When the job completes, you can get the archive contents\n        by calling get_output().\n\n        :param archive: The archive to retrieve.\n        :return: The archive retrieval job.\n        '
        try:
            job = archive.initiate_archive_retrieval()
            logger.info('Started %s job with ID %s.', job.action, job.id)
        except ClientError:
            logger.exception("Couldn't start job on archive %s.", archive.id)
            raise
        else:
            return job

    @staticmethod
    def delete_archive(archive):
        if False:
            i = 10
            return i + 15
        '\n        Deletes an archive from a vault.\n\n        :param archive: The archive to delete.\n        '
        try:
            archive.delete()
            logger.info('Deleted archive %s from vault %s.', archive.id, archive.vault_name)
        except ClientError:
            logger.exception("Couldn't delete archive %s.", archive.id)
            raise

    @staticmethod
    def get_job_status(job):
        if False:
            return 10
        '\n        Gets the status of a job.\n\n        :param job: The job to query.\n        :return: The current status of the job.\n        '
        try:
            job.load()
            logger.info('Job %s is performing action %s and has status %s.', job.id, job.action, job.status_code)
        except ClientError:
            logger.exception("Couldn't get status for job %s.", job.id)
            raise
        else:
            return job.status_code

    @staticmethod
    def get_job_output(job):
        if False:
            while True:
                i = 10
        '\n        Gets the output of a job, such as a vault inventory or the contents of an\n        archive.\n\n        :param job: The job to get output from.\n        :return: The job output, in bytes.\n        '
        try:
            response = job.get_output()
            out_bytes = response['body'].read()
            logger.info('Read %s bytes from job %s.', len(out_bytes), job.id)
            if 'archiveDescription' in response:
                logger.info("These bytes are described as '%s'", response['archiveDescription'])
        except ClientError:
            logger.exception("Couldn't get output for job %s.", job.id)
            raise
        else:
            return out_bytes

    def set_notifications(self, vault, sns_topic_arn):
        if False:
            i = 10
            return i + 15
        '\n        Sets an Amazon Simple Notification Service (Amazon SNS) topic as a target\n        for notifications. Amazon S3 Glacier publishes messages to this topic for\n        the configured list of events.\n\n        :param vault: The vault to set up to publish notifications.\n        :param sns_topic_arn: The Amazon Resource Name (ARN) of the topic that\n                              receives notifications.\n        :return: Data about the new notification configuration.\n        '
        try:
            notification = self.glacier_resource.Notification('-', vault.name)
            notification.set(vaultNotificationConfig={'SNSTopic': sns_topic_arn, 'Events': ['ArchiveRetrievalCompleted', 'InventoryRetrievalCompleted']})
            logger.info('Notifications will be sent to %s for events %s from %s.', notification.sns_topic, notification.events, notification.vault_name)
        except ClientError:
            logger.exception("Couldn't set notifications to %s on %s.", sns_topic_arn, vault.name)
            raise
        else:
            return notification

    @staticmethod
    def get_notification(vault):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets the currently notification configuration for a vault.\n\n        :param vault: The vault to query.\n        :return: The notification configuration for the specified vault.\n        '
        try:
            notification = vault.Notification()
            logger.info('Vault %s notifies %s on %s events.', vault.name, notification.sns_topic, notification.events)
        except ClientError:
            logger.exception("Couldn't get notification data for %s.", vault.name)
            raise
        else:
            return notification

    @staticmethod
    def stop_notifications(notification):
        if False:
            i = 10
            return i + 15
        '\n        Stops notifications to the configured Amazon SNS topic.\n\n        :param notification: The notification configuration to remove.\n        '
        try:
            notification.delete()
            logger.info('Notifications stopped.')
        except ClientError:
            logger.exception("Couldn't stop notifications.")
            raise

def upload_demo(glacier, vault_name, topic_arn):
    if False:
        for i in range(10):
            print('nop')
    '\n    Shows how to:\n    * Create a vault.\n    * Configure the vault to publish notifications to an Amazon SNS topic.\n    * Upload an archive.\n    * Start a job to retrieve the archive.\n\n    :param glacier: A Boto3 Amazon S3 Glacier resource.\n    :param vault_name: The name of the vault to create.\n    :param topic_arn: The ARN of an Amazon SNS topic that receives notification of\n                      Amazon S3 Glacier events.\n    '
    print(f'\nCreating vault {vault_name}.')
    vault = glacier.create_vault(vault_name)
    print('\nList of vaults in your account:')
    glacier.list_vaults()
    print(f'\nUploading glacier_basics.py to {vault.name}.')
    with open('glacier_basics.py', 'rb') as upload_file:
        archive = glacier.upload_archive(vault, 'glacier_basics.py', upload_file)
    print('\nStarting an archive retrieval request to get the file back from the vault.')
    glacier.initiate_archive_retrieval(archive)
    print('\nListing in progress jobs:')
    glacier.list_jobs(vault, 'in_progress')
    print('\nBecause Amazon S3 Glacier is intended for infrequent retrieval, an archive request with Standard retrieval typically completes within 3–5 hours.')
    if topic_arn:
        notification = glacier.set_notifications(vault, topic_arn)
        print(f'\nVault {vault.name} is configured to notify the {notification.sns_topic} topic when {notification.events} events occur. You can subscribe to this topic to receive a message when the archive retrieval completes.\n')
    else:
        print(f'\nVault {vault.name} is not configured to notify an Amazon SNS topic when the archive retrieval completes so wait a few hours.')
    print('\nRetrieve your job output by running this script with the --retrieve flag.')

def retrieve_demo(glacier, vault_name):
    if False:
        return 10
    '\n    Shows how to:\n    * List jobs for a vault and get job status.\n    * Get the output of a completed archive retrieval job.\n    * Delete an archive.\n    * Delete a vault.\n\n    :param glacier: A Boto3 Amazon S3 Glacier resource.\n    :param vault_name: The name of the vault to query for jobs.\n    '
    vault = glacier.glacier_resource.Vault('-', vault_name)
    try:
        vault.load()
    except ClientError as err:
        if err.response['Error']['Code'] == 'ResourceNotFoundException':
            print(f"\nVault {vault_name} doesn't exist. You must first run this script with the --upload flag to create the vault.")
            return
        else:
            raise
    print(f'\nGetting completed jobs for {vault.name}.')
    jobs = glacier.list_jobs(vault, 'completed')
    if not jobs:
        print('\nNo completed jobs found. Give it some time and try again later.')
        return
    retrieval_job = None
    for job in jobs:
        if job.action == 'ArchiveRetrieval' and job.status_code == 'Succeeded':
            retrieval_job = job
            break
    if retrieval_job is None:
        print('\nNo ArchiveRetrieval jobs found. Give it some time and try again later.')
        return
    print(f'\nGetting output from job {retrieval_job.id}.')
    archive_bytes = glacier.get_job_output(retrieval_job)
    archive_str = archive_bytes.decode('utf-8')
    print('\nGot archive data. Printing the first 10 lines.')
    print(os.linesep.join(archive_str.split(os.linesep)[:10]))
    print(f'\nDeleting the archive from {vault.name}.')
    archive = glacier.glacier_resource.Archive('-', vault.name, retrieval_job.archive_id)
    glacier.delete_archive(archive)
    print(f'\nDeleting {vault.name}.')
    glacier.delete_vault(vault)

def usage_demo():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser()
    parser.add_argument('--upload', action='store_true', help='Demonstrates creating a vault, uploading an archive, and starting a retrieval job.')
    parser.add_argument('--notify', help='(Optional) ARN of an Amazon SNS topic that allows Amazon S3 Glacier to publish to it. This is used in the upload demo to set up notifications from Amazon S3 Glacier.')
    parser.add_argument('--retrieve', action='store_true', help='Demonstrates getting job status, retrieving data from an archive, and deleting the archive and vault.')
    args = parser.parse_args()
    print('-' * 88)
    print('Welcome to the Amazon S3 Glacier demo!')
    print('-' * 88)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    vault_name = 'doc-example-vault'
    glacier = GlacierWrapper(boto3.resource('glacier'))
    if args.upload:
        upload_demo(glacier, vault_name, args.notify)
    elif args.retrieve:
        retrieve_demo(glacier, vault_name)
    else:
        parser.print_help()
    print('\nThanks for watching!')
    print('-' * 88)
if __name__ == '__main__':
    usage_demo()