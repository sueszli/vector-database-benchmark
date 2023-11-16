from django.utils.translation import gettext_lazy as _
from .base import BaseType

class GPTTypes(BaseType):
    CHATGPT = ('chatgpt', _('ChatGPT'))

    @classmethod
    def _get_base_constrains(cls) -> dict:
        if False:
            while True:
                i = 10
        return {'*': {'charset_enabled': False, 'domain_enabled': False, 'su_enabled': False}}

    @classmethod
    def _get_automation_constrains(cls) -> dict:
        if False:
            return 10
        constrains = {'*': {'ansible_enabled': False, 'ping_enabled': False, 'gather_facts_enabled': False, 'verify_account_enabled': False, 'change_secret_enabled': False, 'push_account_enabled': False, 'gather_accounts_enabled': False}}
        return constrains

    @classmethod
    def _get_protocol_constrains(cls) -> dict:
        if False:
            i = 10
            return i + 15
        return {'*': {'choices': '__self__'}}

    @classmethod
    def internal_platforms(cls):
        if False:
            return 10
        return {cls.CHATGPT: [{'name': 'ChatGPT'}]}

    @classmethod
    def get_community_types(cls):
        if False:
            i = 10
            return i + 15
        return [cls.CHATGPT]