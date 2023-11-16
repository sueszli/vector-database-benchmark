"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Simple Email Service
(Amazon SES) to manage filters and rules that are applied to incoming email.
"""
import json
import logging
from pprint import pprint
import time
from urllib import request
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class SesReceiptHandler:
    """Encapsulates Amazon SES receipt handling functions."""

    def __init__(self, ses_client, s3_resource):
        if False:
            return 10
        '\n        :param ses_client: A Boto3 Amazon SES client.\n        :param s3_resource: A Boto3 Amazon S3 resource.\n        '
        self.ses_client = ses_client
        self.s3_resource = s3_resource

    def create_receipt_filter(self, filter_name, ip_address_or_range, allow):
        if False:
            while True:
                i = 10
        '\n        Creates a filter that allows or blocks incoming mail from an IP address or\n        range.\n\n        :param filter_name: The name to give the filter.\n        :param ip_address_or_range: The IP address or range to block or allow.\n        :param allow: When True, incoming mail is allowed from the specified IP\n                      address or range; otherwise, it is blocked.\n        '
        try:
            policy = 'Allow' if allow else 'Block'
            self.ses_client.create_receipt_filter(Filter={'Name': filter_name, 'IpFilter': {'Cidr': ip_address_or_range, 'Policy': policy}})
            logger.info('Created receipt filter %s to %s IP of %s.', filter_name, policy, ip_address_or_range)
        except ClientError:
            logger.exception("Couldn't create receipt filter %s.", filter_name)
            raise

    def list_receipt_filters(self):
        if False:
            return 10
        '\n        Gets the list of receipt filters for the current account.\n\n        :return: The list of receipt filters.\n        '
        try:
            response = self.ses_client.list_receipt_filters()
            filters = response['Filters']
            logger.info('Got %s receipt filters.', len(filters))
        except ClientError:
            logger.exception("Couldn't get receipt filters.")
            raise
        else:
            return filters

    def delete_receipt_filter(self, filter_name):
        if False:
            print('Hello World!')
        '\n        Deletes a receipt filter.\n\n        :param filter_name: The name of the filter to delete.\n        '
        try:
            self.ses_client.delete_receipt_filter(FilterName=filter_name)
            logger.info('Deleted receipt filter %s.', filter_name)
        except ClientError:
            logger.exception("Couldn't delete receipt filter %s.", filter_name)
            raise

    def create_receipt_rule_set(self, rule_set_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates an empty rule set. Rule sets contain individual rules and can be\n        used to organize rules.\n\n        :param rule_set_name: The name to give the rule set.\n        '
        try:
            self.ses_client.create_receipt_rule_set(RuleSetName=rule_set_name)
            logger.info('Created receipt rule set %s.', rule_set_name)
        except ClientError:
            logger.exception("Couldn't create receipt rule set %s.", rule_set_name)
            raise

    def create_bucket_for_copy(self, bucket_name):
        if False:
            while True:
                i = 10
        '\n        Creates a bucket that can receive copies of emails from Amazon SES. This\n        includes adding a policy to the bucket that grants Amazon SES permission\n        to put objects in the bucket.\n\n        :param bucket_name: The name of the bucket to create.\n        :return: The newly created bucket.\n        '
        allow_ses_put_policy = {'Version': '2012-10-17', 'Statement': [{'Sid': 'AllowSESPut', 'Effect': 'Allow', 'Principal': {'Service': 'ses.amazonaws.com'}, 'Action': 's3:PutObject', 'Resource': f'arn:aws:s3:::{bucket_name}/*'}]}
        bucket = None
        try:
            bucket = self.s3_resource.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': self.s3_resource.meta.client.meta.region_name})
            bucket.wait_until_exists()
            bucket.Policy().put(Policy=json.dumps(allow_ses_put_policy))
            logger.info('Created bucket %s to receive copies of emails.', bucket_name)
        except ClientError:
            logger.exception("Couldn't create bucket to receive copies of emails.")
            if bucket is not None:
                bucket.delete()
            raise
        else:
            return bucket

    def create_s3_copy_rule(self, rule_set_name, rule_name, recipients, bucket_name, prefix):
        if False:
            i = 10
            return i + 15
        '\n        Creates a rule so that all emails received by the specified recipients are\n        copied to an Amazon S3 bucket.\n\n        :param rule_set_name: The name of a previously created rule set to contain\n                              this rule.\n        :param rule_name: The name to give the rule.\n        :param recipients: When an email is received by one of these recipients, it\n                           is copied to the Amazon S3 bucket.\n        :param bucket_name: The name of the bucket to receive email copies. This\n                            bucket must allow Amazon SES to put objects into it.\n        :param prefix: An object key prefix to give the emails copied to the bucket.\n        '
        try:
            self.ses_client.create_receipt_rule(RuleSetName=rule_set_name, Rule={'Name': rule_name, 'Enabled': True, 'Recipients': recipients, 'Actions': [{'S3Action': {'BucketName': bucket_name, 'ObjectKeyPrefix': prefix}}]})
            logger.info('Created rule %s to copy mail received by %s to bucket %s.', rule_name, recipients, bucket_name)
        except ClientError:
            logger.exception("Couldn't create rule %s.", rule_name)
            raise

    def describe_receipt_rule_set(self, rule_set_name):
        if False:
            return 10
        '\n        Gets data about a rule set.\n\n        :param rule_set_name: The name of the rule set to retrieve.\n        :return: Data about the rule set.\n        '
        try:
            response = self.ses_client.describe_receipt_rule_set(RuleSetName=rule_set_name)
            logger.info('Got data for rule set %s.', rule_set_name)
        except ClientError:
            logger.exception("Couldn't get data for rule set %s.", rule_set_name)
            raise
        else:
            return response

    def delete_receipt_rule(self, rule_set_name, rule_name):
        if False:
            print('Hello World!')
        '\n        Deletes a rule.\n\n        :param rule_set_name: The rule set that contains the rule to delete.\n        :param rule_name: The rule to delete.\n        '
        try:
            self.ses_client.delete_receipt_rule(RuleSetName=rule_set_name, RuleName=rule_name)
            logger.info('Removed rule %s from rule set %s.', rule_name, rule_set_name)
        except ClientError:
            logger.exception("Couldn't remove rule %s from rule set %s.", rule_name, rule_set_name)
            raise

    def delete_receipt_rule_set(self, rule_set_name):
        if False:
            return 10
        '\n        Deletes a rule set. When a rule set is deleted, all of the rules it contains\n        are also deleted.\n\n        :param rule_set_name: The name of the rule set to delete.\n        '
        try:
            self.ses_client.delete_receipt_rule_set(RuleSetName=rule_set_name)
            logger.info('Deleted rule set %s.', rule_set_name)
        except ClientError:
            logger.exception("Couldn't delete rule set %s.", rule_set_name)
            raise

def usage_demo():
    if False:
        return 10
    print('-' * 88)
    print('Welcome to the Amazon Simple Email Service (Amazon SES) receipt rules and filters demo!')
    print('-' * 88)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    ses_receipt = SesReceiptHandler(boto3.client('ses'), boto3.resource('s3'))
    filter_name = 'block-self'
    rule_set_name = 'doc-example-rule-set'
    rule_name = 'copy-mail-to-bucket'
    email = 'example@example.org'
    bucket_name = f'doc-example-bucket-{time.time_ns()}'
    prefix = 'example-emails/'
    current_ip_address = request.urlopen('http://checkip.amazonaws.com').read().decode('utf-8').strip()
    print(f'Adding a filter to block email from the current IP address {current_ip_address}.')
    ses_receipt.create_receipt_filter(filter_name, current_ip_address, False)
    filters = ses_receipt.list_receipt_filters()
    print('Current filters now in effect are:')
    print(*filters, sep='\n')
    print('Removing filter.')
    ses_receipt.delete_receipt_filter(filter_name)
    print(f'Creating a rule set and adding a rule to copy all emails received by {email} to Amazon S3 bucket {bucket_name}.')
    print(f'Creating bucket {bucket_name} to hold emails.')
    bucket = ses_receipt.create_bucket_for_copy(bucket_name)
    ses_receipt.create_receipt_rule_set(rule_set_name)
    ses_receipt.create_s3_copy_rule(rule_set_name, rule_name, [email], bucket.name, prefix)
    rule_set = ses_receipt.describe_receipt_rule_set(rule_set_name)
    print(f'Rule set {rule_set_name} looks like this:')
    pprint(rule_set)
    print(f'Deleting rule {rule_name} and rule set {rule_set_name}.')
    ses_receipt.delete_receipt_rule(rule_set_name, rule_name)
    ses_receipt.delete_receipt_rule_set(rule_set_name)
    print(f'Emptying and deleting bucket {bucket_name}.')
    bucket.objects.delete()
    bucket.delete()
    print('Thanks for watching!')
    print('-' * 88)
if __name__ == '__main__':
    usage_demo()