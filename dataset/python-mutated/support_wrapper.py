"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with AWS Support to
create and manage support cases.
"""
import logging
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class SupportWrapper:
    """Encapsulates Support actions."""

    def __init__(self, support_client):
        if False:
            i = 10
            return i + 15
        '\n        :param support_client: A Boto3 Support client.\n        '
        self.support_client = support_client

    @classmethod
    def from_client(cls):
        if False:
            return 10
        '\n        Instantiates this class from a Boto3 client.\n        '
        support_client = boto3.client('support')
        return cls(support_client)

    def describe_services(self, language):
        if False:
            print('Hello World!')
        '\n        Get the descriptions of AWS services available for support for a language.\n\n        :param language: The language for support services.\n        Currently, only "en" (English) and "ja" (Japanese) are supported.\n        :return: The list of AWS service descriptions.\n        '
        try:
            response = self.support_client.describe_services(language=language)
            services = response['services']
        except ClientError as err:
            if err.response['Error']['Code'] == 'SubscriptionRequiredException':
                logger.info('You must have a Business, Enterprise On-Ramp, or Enterprise Support plan to use the AWS Support API. \n\tPlease upgrade your subscription to run these examples.')
            else:
                logger.error("Couldn't get Support services for language %s. Here's why: %s: %s", language, err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        else:
            return services

    def describe_severity_levels(self, language):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the descriptions of available severity levels for support cases for a language.\n\n        :param language: The language for support severity levels.\n        Currently, only "en" (English) and "ja" (Japanese) are supported.\n        :return: The list of severity levels.\n        '
        try:
            response = self.support_client.describe_severity_levels(language=language)
            severity_levels = response['severityLevels']
        except ClientError as err:
            if err.response['Error']['Code'] == 'SubscriptionRequiredException':
                logger.info('You must have a Business, Enterprise On-Ramp, or Enterprise Support plan to use the AWS Support API. \n\tPlease upgrade your subscription to run these examples.')
            else:
                logger.error("Couldn't get severity levels for language %s. Here's why: %s: %s", language, err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        else:
            return severity_levels

    def create_case(self, service, category, severity):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new support case.\n\n        :param service: The service to use for the new case.\n        :param category: The category to use for the new case.\n        :param severity: The severity to use for the new case.\n        :return: The caseId of the new case.\n        '
        try:
            response = self.support_client.create_case(subject='Example case for testing, ignore.', serviceCode=service['code'], severityCode=severity['code'], categoryCode=category['code'], communicationBody='Example support case body.', language='en', issueType='customer-service')
            case_id = response['caseId']
        except ClientError as err:
            if err.response['Error']['Code'] == 'SubscriptionRequiredException':
                logger.info('You must have a Business, Enterprise On-Ramp, or Enterprise Support plan to use the AWS Support API. \n\tPlease upgrade your subscription to run these examples.')
            else:
                logger.error("Couldn't create case. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        else:
            return case_id

    def add_attachment_to_set(self):
        if False:
            print('Hello World!')
        '\n        Add an attachment to a set, or create a new attachment set if one does not exist.\n\n        :return: The attachment set ID.\n        '
        try:
            response = self.support_client.add_attachments_to_set(attachments=[{'fileName': 'attachment_file.txt', 'data': b'This is a sample file for attachment to a support case.'}])
            new_set_id = response['attachmentSetId']
        except ClientError as err:
            if err.response['Error']['Code'] == 'SubscriptionRequiredException':
                logger.info('You must have a Business, Enterprise On-Ramp, or Enterprise Support plan to use the AWS Support API. \n\tPlease upgrade your subscription to run these examples.')
            else:
                logger.error("Couldn't add attachment. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        else:
            return new_set_id

    def add_communication_to_case(self, attachment_set_id, case_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a communication and an attachment set to a case.\n\n        :param attachment_set_id: The ID of an existing attachment set.\n        :param case_id: The ID of the case.\n        '
        try:
            self.support_client.add_communication_to_case(caseId=case_id, communicationBody='This is an example communication added to a support case.', attachmentSetId=attachment_set_id)
        except ClientError as err:
            if err.response['Error']['Code'] == 'SubscriptionRequiredException':
                logger.info('You must have a Business, Enterprise On-Ramp, or Enterprise Support plan to use the AWS Support API. \n\tPlease upgrade your subscription to run these examples.')
            else:
                logger.error("Couldn't add communication. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
                raise

    def describe_all_case_communications(self, case_id):
        if False:
            print('Hello World!')
        '\n        Describe all the communications for a case using a paginator.\n\n        :param case_id: The ID of the case.\n        :return: The communications for the case.\n        '
        try:
            communications = []
            paginator = self.support_client.get_paginator('describe_communications')
            for page in paginator.paginate(caseId=case_id):
                communications += page['communications']
        except ClientError as err:
            if err.response['Error']['Code'] == 'SubscriptionRequiredException':
                logger.info('You must have a Business, Enterprise On-Ramp, or Enterprise Support plan to use the AWS Support API. \n\tPlease upgrade your subscription to run these examples.')
            else:
                logger.error("Couldn't describe communications. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        else:
            return communications

    def describe_attachment(self, attachment_id):
        if False:
            return 10
        '\n        Get information about an attachment by its attachmentID.\n\n        :param attachment_id: The ID of the attachment.\n        :return: The name of the attached file.\n        '
        try:
            response = self.support_client.describe_attachment(attachmentId=attachment_id)
            attached_file = response['attachment']['fileName']
        except ClientError as err:
            if err.response['Error']['Code'] == 'SubscriptionRequiredException':
                logger.info('You must have a Business, Enterprise On-Ramp, or Enterprise Support plan to use the AWS Support API. \n\tPlease upgrade your subscription to run these examples.')
            else:
                logger.error("Couldn't get attachment description. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        else:
            return attached_file

    def resolve_case(self, case_id):
        if False:
            return 10
        '\n        Resolve a support case by its caseId.\n\n        :param case_id: The ID of the case to resolve.\n        :return: The final status of the case.\n        '
        try:
            response = self.support_client.resolve_case(caseId=case_id)
            final_status = response['finalCaseStatus']
        except ClientError as err:
            if err.response['Error']['Code'] == 'SubscriptionRequiredException':
                logger.info('You must have a Business, Enterprise On-Ramp, or Enterprise Support plan to use the AWS Support API. \n\tPlease upgrade your subscription to run these examples.')
            else:
                logger.error("Couldn't resolve case. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        else:
            return final_status

    def describe_cases(self, after_time, before_time, resolved):
        if False:
            i = 10
            return i + 15
        '\n        Describe support cases over a period of time, optionally filtering\n        by status.\n\n        :param after_time: The start time to include for cases.\n        :param before_time: The end time to include for cases.\n        :param resolved: True to include resolved cases in the results,\n            otherwise results are open cases.\n        :return: The final status of the case.\n        '
        try:
            cases = []
            paginator = self.support_client.get_paginator('describe_cases')
            for page in paginator.paginate(afterTime=after_time, beforeTime=before_time, includeResolvedCases=resolved, language='en'):
                cases += page['cases']
        except ClientError as err:
            if err.response['Error']['Code'] == 'SubscriptionRequiredException':
                logger.info('You must have a Business, Enterprise On-Ramp, or Enterprise Support plan to use the AWS Support API. \n\tPlease upgrade your subscription to run these examples.')
            else:
                logger.error("Couldn't describe cases. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        else:
            if resolved:
                cases = filter(lambda case: case['status'] == 'resolved', cases)
            return cases