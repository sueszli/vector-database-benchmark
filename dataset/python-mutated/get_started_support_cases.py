"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with AWS Support to
do the following:

1.  Get and display services. Select a service from the list.
2.  Select a category from the selected service.
3.  Get and display severity levels and select a severity level from the list.
4.  Create a support case using the selected service, category, and severity level.
5.  Get and display a list of open support cases for the current day.
6.  Create an attachment set with a sample text file to add to the case.
7.  Add a communication with the attachment to the support case.
8.  List the communications of the support case.
9.  Describe the attachment set.
10. Resolve the support case.
11. Get a list of resolved cases for the current day.
"""
from datetime import datetime, timedelta
import logging
import sys
import boto3
from botocore.exceptions import ClientError
from support_wrapper import SupportWrapper
sys.path.append('../..')
import demo_tools.question as q
from demo_tools.retries import wait
logger = logging.getLogger(__name__)

class SupportCasesScenario:
    """Runs an interactive scenario that shows how to get started using AWS Support."""

    def __init__(self, support_wrapper):
        if False:
            print('Hello World!')
        '\n        :param support_wrapper: An object that wraps AWS Support actions.\n        '
        self.support_wrapper = support_wrapper

    def display_and_select_service(self):
        if False:
            print('Hello World!')
        '\n        Lists support services and prompts the user to select one.\n\n        :return: The support service selected by the user.\n        '
        print('-' * 88)
        services_list = self.support_wrapper.describe_services('en')
        print(f'AWS Support client returned {len(services_list)} services.')
        print('Displaying first 10 services:')
        service_choices = [svc['name'] for svc in services_list[:10]]
        selected_index = q.choose('Select an example support service by entering a number from the preceding list:', service_choices)
        selected_service = services_list[selected_index]
        print('-' * 88)
        return selected_service

    def display_and_select_category(self, service):
        if False:
            i = 10
            return i + 15
        '\n        Lists categories for a support service and prompts the user to select one.\n\n        :param service: The service of the categories.\n        :return: The selected category.\n        '
        print('-' * 88)
        print(f"Available support categories for Service {service['name']} {len(service['categories'])}:")
        categories_choices = [category['name'] for category in service['categories']]
        selected_index = q.choose('Select an example support category by entering a number from the preceding list:', categories_choices)
        selected_category = service['categories'][selected_index]
        print('-' * 88)
        return selected_category

    def display_and_select_severity(self):
        if False:
            while True:
                i = 10
        '\n        Lists available severity levels and prompts the user to select one.\n\n        :return: The selected severity level.\n        '
        print('-' * 88)
        severity_levels_list = self.support_wrapper.describe_severity_levels('en')
        print(f'Available severity levels:')
        severity_choices = [level['name'] for level in severity_levels_list]
        selected_index = q.choose('Select an example severity level by entering a number from the preceding list:', severity_choices)
        selected_severity = severity_levels_list[selected_index]
        print('-' * 88)
        return selected_severity

    def create_example_case(self, service, category, severity_level):
        if False:
            while True:
                i = 10
        "\n        Creates an example support case with the user's selections.\n\n        :param service: The service for the new case.\n        :param category: The category for the new case.\n        :param severity_level: The severity level for the new case.\n        :return: The caseId of the new support case.\n        "
        print('-' * 88)
        print(f"Creating new case for service {service['name']}.")
        case_id = self.support_wrapper.create_case(service, category, severity_level)
        print(f'\tNew case created with ID {case_id}.')
        print('-' * 88)
        return case_id

    def list_open_cases(self):
        if False:
            while True:
                i = 10
        '\n        List the open cases for the current day.\n        '
        print('-' * 88)
        print("Let's list the open cases for the current day.")
        start_time = str(datetime.utcnow().date())
        end_time = str(datetime.utcnow().date() + timedelta(days=1))
        open_cases = self.support_wrapper.describe_cases(start_time, end_time, False)
        for case in open_cases:
            print(f"\tCase: {case['caseId']}: status {case['status']}.")
        print('-' * 88)

    def create_attachment_set(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create an attachment set with a sample file.\n\n        :return: The attachment set ID of the new attachment set.\n        '
        print('-' * 88)
        print('Creating attachment set with a sample file.')
        attachment_set_id = self.support_wrapper.add_attachment_to_set()
        print(f'\tNew attachment set created with ID {attachment_set_id}.')
        print('-' * 88)
        return attachment_set_id

    def add_communication(self, case_id, attachment_set_id):
        if False:
            print('Hello World!')
        '\n        Add a communication with an attachment set to the case.\n\n        :param case_id: The ID of the case for the communication.\n        :param attachment_set_id: The ID of the attachment set to\n        add to the communication.\n        '
        print('-' * 88)
        print(f'Adding a communication and attachment set to the case.')
        self.support_wrapper.add_communication_to_case(attachment_set_id, case_id)
        print(f'Added a communication and attachment set {attachment_set_id} to the case {case_id}.')
        print('-' * 88)

    def list_communications(self, case_id):
        if False:
            while True:
                i = 10
        '\n        List the communications associated with a case.\n\n        :param case_id: The ID of the case.\n        :return: The attachment ID of an attachment.\n        '
        print('-' * 88)
        print("Let's list the communications for our case.")
        attachment_id = ''
        communications = self.support_wrapper.describe_all_case_communications(case_id)
        for communication in communications:
            print(f"\tCommunication created on {communication['timeCreated']} has {len(communication['attachmentSet'])} attachments.")
            if len(communication['attachmentSet']) > 0:
                attachment_id = communication['attachmentSet'][0]['attachmentId']
        print('-' * 88)
        return attachment_id

    def describe_case_attachment(self, attachment_id):
        if False:
            while True:
                i = 10
        '\n        Describe an attachment associated with a case.\n\n        :param attachment_id: The ID of the attachment.\n        '
        print('-' * 88)
        print("Let's list the communications for our case.")
        attached_file = self.support_wrapper.describe_attachment(attachment_id)
        print(f'\tAttachment includes file {attached_file}.')
        print('-' * 88)

    def resolve_case(self, case_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Shows how to resolve an AWS Support case by its ID.\n\n        :param case_id: The ID of the case to resolve.\n        '
        print('-' * 88)
        print(f'Resolving case with ID {case_id}.')
        case_status = self.support_wrapper.resolve_case(case_id)
        print(f'\tFinal case status is {case_status}.')
        print('-' * 88)

    def list_resolved_cases(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        List the resolved cases for the current day.\n        '
        print('-' * 88)
        print("Let's list the resolved cases for the current day.")
        start_time = str(datetime.utcnow().date())
        end_time = str(datetime.utcnow().date() + timedelta(days=1))
        resolved_cases = self.support_wrapper.describe_cases(start_time, end_time, True)
        for case in resolved_cases:
            print(f"\tCase: {case['caseId']}: status {case['status']}.")
        print('-' * 88)

    def run_scenario(self):
        if False:
            for i in range(10):
                print('nop')
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        print('-' * 88)
        print('Welcome to the AWS Support get started with support cases demo.')
        print('-' * 88)
        selected_service = self.display_and_select_service()
        selected_category = self.display_and_select_category(selected_service)
        selected_severity = self.display_and_select_severity()
        new_case_id = self.create_example_case(selected_service, selected_category, selected_severity)
        wait(10)
        self.list_open_cases()
        new_attachment_set_id = self.create_attachment_set()
        self.add_communication(new_case_id, new_attachment_set_id)
        new_attachment_id = self.list_communications(new_case_id)
        self.describe_case_attachment(new_attachment_id)
        self.resolve_case(new_case_id)
        wait(10)
        self.list_resolved_cases()
        print('\nThanks for watching!')
        print('-' * 88)
if __name__ == '__main__':
    try:
        scenario = SupportCasesScenario(SupportWrapper.from_client())
        scenario.run_scenario()
    except Exception:
        logging.exception('Something went wrong with the demo.')