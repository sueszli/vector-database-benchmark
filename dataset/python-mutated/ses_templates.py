"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Simple Email Service
(Amazon SES) to manage email templates that contain replaceable tags.
"""
import logging
from pprint import pprint
import re
import boto3
from botocore.exceptions import ClientError
TEMPLATE_REGEX = '(?<={{).+?(?=}})'
logger = logging.getLogger(__name__)

class SesTemplate:
    """Encapsulates Amazon SES template functions."""

    def __init__(self, ses_client):
        if False:
            while True:
                i = 10
        '\n        :param ses_client: A Boto3 Amazon SES client.\n        '
        self.ses_client = ses_client
        self.template = None
        self.template_tags = set()

    def _extract_tags(self, subject, text, html):
        if False:
            return 10
        '\n        Extracts tags from a template as a set of unique values.\n\n        :param subject: The subject of the email.\n        :param text: The text version of the email.\n        :param html: The html version of the email.\n        '
        self.template_tags = set(re.findall(TEMPLATE_REGEX, subject + text + html))
        logger.info('Extracted template tags: %s', self.template_tags)

    def verify_tags(self, template_data):
        if False:
            i = 10
            return i + 15
        '\n        Verifies that the tags in the template data are part of the template.\n\n        :param template_data: Template data formed of key-value pairs of tags and\n                              replacement text.\n        :return: True when all of the tags in the template data are usable with the\n                 template; otherwise, False.\n        '
        diff = set(template_data) - self.template_tags
        if diff:
            logger.warning("Template data contains tags that aren't in the template: %s", diff)
            return False
        else:
            return True

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :return: Gets the name of the template, if a template has been loaded.\n        '
        return self.template['TemplateName'] if self.template is not None else None

    def create_template(self, name, subject, text, html):
        if False:
            while True:
                i = 10
        '\n        Creates an email template.\n\n        :param name: The name of the template.\n        :param subject: The subject of the email.\n        :param text: The plain text version of the email.\n        :param html: The HTML version of the email.\n        '
        try:
            template = {'TemplateName': name, 'SubjectPart': subject, 'TextPart': text, 'HtmlPart': html}
            self.ses_client.create_template(Template=template)
            logger.info('Created template %s.', name)
            self.template = template
            self._extract_tags(subject, text, html)
        except ClientError:
            logger.exception("Couldn't create template %s.", name)
            raise

    def delete_template(self):
        if False:
            i = 10
            return i + 15
        '\n        Deletes an email template.\n        '
        try:
            self.ses_client.delete_template(TemplateName=self.template['TemplateName'])
            logger.info('Deleted template %s.', self.template['TemplateName'])
            self.template = None
            self.template_tags = None
        except ClientError:
            logger.exception("Couldn't delete template %s.", self.template['TemplateName'])
            raise

    def get_template(self, name):
        if False:
            return 10
        '\n        Gets a previously created email template.\n\n        :param name: The name of the template to retrieve.\n        :return: The retrieved email template.\n        '
        try:
            response = self.ses_client.get_template(TemplateName=name)
            self.template = response['Template']
            logger.info('Got template %s.', name)
            self._extract_tags(self.template['SubjectPart'], self.template['TextPart'], self.template['HtmlPart'])
        except ClientError:
            logger.exception("Couldn't get template %s.", name)
            raise
        else:
            return self.template

    def list_templates(self):
        if False:
            i = 10
            return i + 15
        '\n        Gets a list of all email templates for the current account.\n\n        :return: The list of retrieved email templates.\n        '
        try:
            response = self.ses_client.list_templates()
            templates = response['TemplatesMetadata']
            logger.info('Got %s templates.', len(templates))
        except ClientError:
            logger.exception("Couldn't get templates.")
            raise
        else:
            return templates

    def update_template(self, name, subject, text, html):
        if False:
            print('Hello World!')
        '\n        Updates a previously created email template.\n\n        :param name: The name of the template.\n        :param subject: The subject of the email.\n        :param text: The plain text version of the email.\n        :param html: The HTML version of the email.\n        '
        try:
            template = {'TemplateName': name, 'SubjectPart': subject, 'TextPart': text, 'HtmlPart': html}
            self.ses_client.update_template(Template=template)
            logger.info('Updated template %s.', name)
            self.template = template
            self._extract_tags(subject, text, html)
        except ClientError:
            logger.exception("Couldn't update template %s.", name)
            raise

def usage_demo():
    if False:
        while True:
            i = 10
    print('-' * 88)
    print('Welcome to the Amazon Simple Email Service (Amazon SES) email template demo!')
    print('-' * 88)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    ses_template = SesTemplate(boto3.client('ses'))
    template = {'name': 'doc-example-template', 'subject': 'Example of an email template.', 'text': "This is what {{name}} will {{action}} if {{name}} can't display HTML.", 'html': '<p><i>This</i> is what {{name}} will {{action}} if {{name}} <b>can</b> display HTML.</p>'}
    print('Creating an email template.')
    ses_template.create_template(**template)
    print('Getting the list of template metadata.')
    template_metas = ses_template.list_templates()
    for temp_meta in template_metas:
        print(f"Got template {temp_meta['Name']}:")
        temp_data = ses_template.get_template(temp_meta['Name'])
        pprint(temp_data)
    print(f"Deleting template {template['name']}.")
    ses_template.delete_template()
    print('Thanks for watching!')
    print('-' * 88)
if __name__ == '__main__':
    usage_demo()