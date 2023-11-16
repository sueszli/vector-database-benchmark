import logging
from string import Template
import requests
from redash.destinations import BaseDestination, register
from redash.utils import json_dumps

def json_string_substitute(j, substitutions):
    if False:
        while True:
            i = 10
    '\n    Alternative to string.format when the string has braces.\n    :param j: json string that will have substitutions\n    :type j: str\n    :param substitutions: dictionary of values to be replaced\n    :type substitutions: dict\n    '
    if substitutions:
        substitution_candidate = j.replace('{', '${')
        string_template = Template(substitution_candidate)
        substituted = string_template.safe_substitute(substitutions)
        out_str = substituted.replace('${', '{')
        return out_str
    else:
        return j

class MicrosoftTeamsWebhook(BaseDestination):
    ALERTS_DEFAULT_MESSAGE_TEMPLATE = json_dumps({'@type': 'MessageCard', '@context': 'http://schema.org/extensions', 'themeColor': '0076D7', 'summary': 'A Redash Alert was Triggered', 'sections': [{'activityTitle': 'A Redash Alert was Triggered', 'facts': [{'name': 'Alert Name', 'value': '{alert_name}'}, {'name': 'Alert URL', 'value': '{alert_url}'}, {'name': 'Query', 'value': '{query_text}'}, {'name': 'Query URL', 'value': '{query_url}'}], 'markdown': True}]})

    @classmethod
    def name(cls):
        if False:
            return 10
        return 'Microsoft Teams Webhook'

    @classmethod
    def type(cls):
        if False:
            print('Hello World!')
        return 'microsoft_teams_webhook'

    @classmethod
    def configuration_schema(cls):
        if False:
            for i in range(10):
                print('nop')
        return {'type': 'object', 'properties': {'url': {'type': 'string', 'title': 'Microsoft Teams Webhook URL'}, 'message_template': {'type': 'string', 'default': MicrosoftTeamsWebhook.ALERTS_DEFAULT_MESSAGE_TEMPLATE, 'title': 'Message Template'}}, 'required': ['url']}

    @classmethod
    def icon(cls):
        if False:
            while True:
                i = 10
        return 'fa-bolt'

    def notify(self, alert, query, user, new_state, app, host, metadata, options):
        if False:
            while True:
                i = 10
        '\n        :type app: redash.Redash\n        '
        try:
            alert_url = '{host}/alerts/{alert_id}'.format(host=host, alert_id=alert.id)
            query_url = '{host}/queries/{query_id}'.format(host=host, query_id=query.id)
            message_template = options.get('message_template', MicrosoftTeamsWebhook.ALERTS_DEFAULT_MESSAGE_TEMPLATE)
            payload = json_string_substitute(message_template, {'alert_name': alert.name, 'alert_url': alert_url, 'query_text': query.query_text, 'query_url': query_url})
            headers = {'Content-Type': 'application/json'}
            resp = requests.post(options.get('url'), data=payload, headers=headers, timeout=5.0)
            if resp.status_code != 200:
                logging.error('MS Teams Webhook send ERROR. status_code => {status}'.format(status=resp.status_code))
        except Exception:
            logging.exception('MS Teams Webhook send ERROR.')
register(MicrosoftTeamsWebhook)