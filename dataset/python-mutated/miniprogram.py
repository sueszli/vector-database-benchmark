from wechatpy.client.api.base import BaseWeChatAPI

class WeChatMiniProgram(BaseWeChatAPI):
    """
    小程序接口（服务商、第三方应用开发相关）

    https://work.weixin.qq.com/api/doc/90001/90144/92423

    新的授权体系有部分接口未实现，欢迎提交 PR。
    """

    def jscode2session(self, js_code):
        if False:
            return 10
        '\n        临时登录凭证校验接口\n\n        详情请参考\n        https://work.weixin.qq.com/api/doc/90001/90143/90603\n        :param js_code: 登录时获取的 code\n        :return: 返回的 JSON 数据包\n        '
        return self._get('service/miniprogram/jscode2session', params={'js_code': js_code, 'grant_type': 'authorization_code'})