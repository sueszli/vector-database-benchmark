from collections import OrderedDict
from django.conf import settings
from common.exceptions import JMSException
from common.utils import get_logger
from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.sms.v20210111 import sms_client, models
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from .base import BaseSMSClient
logger = get_logger(__file__)

class TencentSMS(BaseSMSClient):
    """
    https://cloud.tencent.com/document/product/382/43196#.E5.8F.91.E9.80.81.E7.9F.AD.E4.BF.A1
    """
    SIGN_AND_TMPL_SETTING_FIELD_PREFIX = 'TENCENT'

    @classmethod
    def new_from_settings(cls):
        if False:
            for i in range(10):
                print('nop')
        return cls(secret_id=settings.TENCENT_SECRET_ID, secret_key=settings.TENCENT_SECRET_KEY, sdkappid=settings.TENCENT_SDKAPPID)

    def __init__(self, secret_id: str, secret_key: str, sdkappid: str):
        if False:
            i = 10
            return i + 15
        self.sdkappid = sdkappid
        cred = credential.Credential(secret_id, secret_key)
        httpProfile = HttpProfile()
        httpProfile.reqMethod = 'POST'
        httpProfile.reqTimeout = 30
        httpProfile.endpoint = 'sms.tencentcloudapi.com'
        clientProfile = ClientProfile()
        clientProfile.signMethod = 'TC3-HMAC-SHA256'
        clientProfile.language = 'en-US'
        clientProfile.httpProfile = httpProfile
        self.client = sms_client.SmsClient(cred, 'ap-guangzhou', clientProfile)

    def send_sms(self, phone_numbers: list, sign_name: str, template_code: str, template_param: OrderedDict, **kwargs):
        if False:
            return 10
        try:
            req = models.SendSmsRequest()
            req.SmsSdkAppId = self.sdkappid
            req.SignName = sign_name
            req.ExtendCode = ''
            req.SessionContext = 'Jumpserver'
            req.SenderId = ''
            req.PhoneNumberSet = phone_numbers
            req.TemplateId = template_code
            req.TemplateParamSet = list(template_param.values())
            logger.info(f'Tencent sms send: phone_numbers={phone_numbers} sign_name={sign_name} template_code={template_code} template_param={template_param}')
            resp = self.client.SendSms(req)
            try:
                code = resp.SendStatusSet[0].Code
                msg = resp.SendStatusSet[0].Message
            except IndexError:
                raise JMSException(code='response_bad', detail=resp)
            if code.lower() != 'ok':
                raise JMSException(code=code, detail=msg)
            return resp
        except TencentCloudSDKException as e:
            raise JMSException(code=e.code, detail=e.message)
client = TencentSMS