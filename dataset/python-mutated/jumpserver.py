from django.core.cache import cache
from django.shortcuts import reverse, redirect
from django.utils.translation import gettext_noop
from .random import random_string
__all__ = ['FlashMessageUtil']

class FlashMessageUtil:
    """
    跳转到通用msg页面
    message_data: {
        'title': '',
        'message': '',
        'error': '',
        'redirect_url': '',
        'confirm_button': '',
        'cancel_url': ''
    }
    """

    @staticmethod
    def get_key(code):
        if False:
            while True:
                i = 10
        key = 'MESSAGE_{}'.format(code)
        return key

    @classmethod
    def get_message_code(cls, message_data):
        if False:
            while True:
                i = 10
        code = random_string(12)
        key = cls.get_key(code)
        cache.set(key, message_data, 60)
        return code

    @classmethod
    def get_message_by_code(cls, code):
        if False:
            i = 10
            return i + 15
        key = cls.get_key(code)
        return cache.get(key)

    @classmethod
    def gen_message_url(cls, message_data):
        if False:
            print('Hello World!')
        code = cls.get_message_code(message_data)
        return reverse('common:flash-message') + f'?code={code}'

    @classmethod
    def gen_and_redirect_to(cls, message_data):
        if False:
            return 10
        url = cls.gen_message_url(message_data)
        return redirect(url)