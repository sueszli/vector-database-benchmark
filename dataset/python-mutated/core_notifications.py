"""Core set of Notifications as a Plugin."""
from django.template.loader import render_to_string
from django.utils.translation import gettext_lazy as _
import requests
from allauth.account.models import EmailAddress
import common.models
import InvenTree.email
import InvenTree.helpers
import InvenTree.tasks
from plugin import InvenTreePlugin, registry
from plugin.mixins import BulkNotificationMethod, SettingsContentMixin, SettingsMixin

class PlgMixin:
    """Mixin to access plugin easier.

    This needs to be spit out to reference the class. Perks of python.
    """

    def get_plugin(self):
        if False:
            while True:
                i = 10
        'Return plugin reference.'
        return InvenTreeCoreNotificationsPlugin

class InvenTreeCoreNotificationsPlugin(SettingsContentMixin, SettingsMixin, InvenTreePlugin):
    """Core notification methods for InvenTree."""
    NAME = 'InvenTreeCoreNotificationsPlugin'
    TITLE = _('InvenTree Notifications')
    AUTHOR = _('InvenTree contributors')
    DESCRIPTION = _('Integrated outgoing notification methods')
    VERSION = '1.0.0'
    SETTINGS = {'ENABLE_NOTIFICATION_EMAILS': {'name': _('Enable email notifications'), 'description': _('Allow sending of emails for event notifications'), 'default': False, 'validator': bool}, 'ENABLE_NOTIFICATION_SLACK': {'name': _('Enable slack notifications'), 'description': _('Allow sending of slack channel messages for event notifications'), 'default': False, 'validator': bool}, 'NOTIFICATION_SLACK_URL': {'name': _('Slack incoming webhook url'), 'description': _('URL that is used to send messages to a slack channel'), 'protected': True}}

    def get_settings_content(self, request):
        if False:
            while True:
                i = 10
        'Custom settings content for the plugin.'
        return '\n        <p>Setup for Slack:</p>\n        <ol>\n            <li>Create a new Slack app on <a href="https://api.slack.com/apps/new" target="_blank">this page</a></li>\n            <li>Enable <i>Incoming Webhooks</i> for the channel you want the notifications posted to</li>\n            <li>Set the webhook URL in the settings above</li>\n        <li>Enable the plugin</li>\n        '

    class EmailNotification(PlgMixin, BulkNotificationMethod):
        """Notificationmethod for delivery via Email."""
        METHOD_NAME = 'mail'
        METHOD_ICON = 'fa-envelope'
        CONTEXT_EXTRA = [('template',), ('template', 'html'), ('template', 'subject')]
        GLOBAL_SETTING = 'ENABLE_NOTIFICATION_EMAILS'
        USER_SETTING = {'name': _('Enable email notifications'), 'description': _('Allow sending of emails for event notifications'), 'default': True, 'validator': bool}

        def get_targets(self):
            if False:
                print('Hello World!')
            'Return a list of target email addresses, only for users which allow email notifications.'
            allowed_users = []
            for user in self.targets:
                if not user.is_active:
                    continue
                allows_emails = InvenTree.helpers.str2bool(self.usersetting(user))
                if allows_emails:
                    allowed_users.append(user)
            return EmailAddress.objects.filter(user__in=allowed_users)

        def send_bulk(self):
            if False:
                while True:
                    i = 10
            'Send the notifications out via email.'
            html_message = render_to_string(self.context['template']['html'], self.context)
            targets = self.targets.values_list('email', flat=True)
            instance_title = common.models.InvenTreeSetting.get_setting('INVENTREE_INSTANCE')
            subject = self.context['template'].get('subject', '')
            if instance_title:
                subject = f'[{instance_title}] {subject}'
            InvenTree.email.send_email(subject, '', targets, html_message=html_message)
            return True

    class SlackNotification(PlgMixin, BulkNotificationMethod):
        """Notificationmethod for delivery via Slack channel messages."""
        METHOD_NAME = 'slack'
        METHOD_ICON = 'fa-envelope'
        GLOBAL_SETTING = 'ENABLE_NOTIFICATION_SLACK'

        def get_targets(self):
            if False:
                i = 10
                return i + 15
            'Not used by this method.'
            return self.targets

        def send_bulk(self):
            if False:
                return 10
            'Send the notifications out via slack.'
            instance = registry.plugins.get(self.get_plugin().NAME.lower())
            url = instance.get_setting('NOTIFICATION_SLACK_URL')
            if not url:
                return False
            ret = requests.post(url, json={'text': str(self.context['message']), 'blocks': [{'type': 'section', 'text': {'type': 'plain_text', 'text': str(self.context['name'])}}, {'type': 'section', 'text': {'type': 'mrkdwn', 'text': str(self.context['message'])}, 'accessory': {'type': 'button', 'text': {'type': 'plain_text', 'text': str(_('Open link')), 'emoji': True}, 'value': f'{self.category}_{self.obj.pk}', 'url': self.context['link'], 'action_id': 'button-action'}}]})
            return ret.ok