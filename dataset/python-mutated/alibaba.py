import json
from django.utils.translation import gettext_lazy as _
from django.conf import settings
from alibabacloud_dysmsapi20170525.client import Client as Dysmsapi20170525Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_dysmsapi20170525 import models as dysmsapi_20170525_models
from Tea.exceptions import TeaException
from common.utils import get_logger
from common.exceptions import JMSException
from .base import BaseSMSClient
logger = get_logger(__file__)

class AlibabaSMS(BaseSMSClient):
    SIGN_AND_TMPL_SETTING_FIELD_PREFIX = 'ALIBABA'

    @classmethod
    def new_from_settings(cls):
        if False:
            print('Hello World!')
        return cls(access_key_id=settings.ALIBABA_ACCESS_KEY_ID, access_key_secret=settings.ALIBABA_ACCESS_KEY_SECRET)

    def __init__(self, access_key_id: str, access_key_secret: str):
        if False:
            for i in range(10):
                print('nop')
        config = open_api_models.Config(access_key_id=access_key_id, access_key_secret=access_key_secret)
        config.endpoint = 'dysmsapi.aliyuncs.com'
        self.client = Dysmsapi20170525Client(config)

    def send_sms(self, phone_numbers: list, sign_name: str, template_code: str, template_param: dict, **kwargs):
        if False:
            i = 10
            return i + 15
        phone_numbers_str = ','.join(phone_numbers)
        send_sms_request = dysmsapi_20170525_models.SendSmsRequest(phone_numbers=phone_numbers_str, sign_name=sign_name, template_code=template_code, template_param=json.dumps(template_param))
        try:
            logger.info(f'Alibaba sms send: phone_numbers={phone_numbers} sign_name={sign_name} template_code={template_code} template_param={template_param}')
            response = self.client.send_sms(send_sms_request)
            if response.body.code != 'OK':
                raise JMSException(detail=response.body.message, code=response.body.code)
        except TeaException as e:
            if e.code == 'SignatureDoesNotMatch':
                raise JMSException(code=e.code, detail=_('Signature does not match'))
            raise JMSException(code=e.code, detail=e.message)
        return response
client = AlibabaSMS