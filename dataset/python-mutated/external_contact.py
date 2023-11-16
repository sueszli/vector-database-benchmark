from typing import Optional, List, Iterator
from optionaldict import optionaldict
from wechatpy.client.api.base import BaseWeChatAPI

class WeChatExternalContact(BaseWeChatAPI):
    """
    外部联系人管理

    详细说明请查阅企业微信有关 `外部联系人管理`_ 的文档。

    .. _外部联系人管理: https://work.weixin.qq.com/api/doc/90000/90135/92109

    .. _客户联系secret: https://work.weixin.qq.com/api/doc/90000/90135/92570
            #13473/%E5%BC%80%E5%A7%8B%E5%BC%80%E5%8F%91

    .. _可调用应用: https://work.weixin.qq.com/api/doc/90000/90135/92570#134
        73/%E5%BC%80%E5%A7%8B%E5%BC%80%E5%8F%91

    .. _客户联系功能: https://work.weixin.qq.com/api/doc/90000/90135/92125
        #13473/%E5%BC%80%E5%A7%8B%E5%BC%80%E5%8F%91

    .. _企业客户权限: https://work.weixin.qq.com/api/doc/90000/90135/92572#19519

    .. _获取外部联系人详情: https://work.weixin.qq.com/api/doc/90000/90135/92572
        #13878
    """

    def get_follow_user_list(self) -> dict:
        if False:
            print('Hello World!')
        '\n        获取配置了客户联系功能的成员列表\n\n        企业和第三方服务商可获取配置了客户联系功能的成员列表。\n\n        详细请查阅企业微信官方文档 `获取配置了客户联系功能的成员列表`_ 章节。\n\n        使用示例:\n\n        .. code-block:: python\n\n            from wechatpy.work import WeChatClient\n\n            # 需要注意使用正确的secret，否则会导致在之后的接口调用中失败\n            client = WeChatClient("corp_id", "secret_key")\n            # 获取成员用户userid列表数据\n            follow_users = client.external_contact.get_follow_user_list()["follow_user"]\n\n        :return: 配置了客户联系功能的成员用户userid信息\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 第三方应用需拥有“企业客户”权限。\n            - 第三方/自建应用只能获取到可见范围内的配置了客户联系功能的成员。\n\n        .. _获取配置了客户联系功能的成员列表:\n            https://work.weixin.qq.com/api/doc/90000/90135/92570\n        '
        return self._get('externalcontact/get_follow_user_list')

    def list(self, userid: str) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        获取客户列表\n\n        企业可通过此接口获取指定成员添加的客户列表。客户是指配置了客户联系功能的成员所添加的\n        外部联系人。没有配置客户联系功能的成员，所添加的外部联系人将不会作为客户返回。\n\n        详细请查阅企业微信官方文档 `获取客户列表`_ 章节。\n\n        使用示例:\n\n        .. code-block:: python\n\n            from wechatpy.work import WeChatClient\n\n            # 需要注意使用正确的secret，否则会导致在之后的接口调用中失败\n            client = WeChatClient("corp_id", "secret_key")\n            # 获取外部联系人的userid列表\n            follow_users = client.external_contact.list("user_id")["external_userid"]\n\n        :param userid: 企业成员的userid\n        :return: 包含外部联系人的userid列表的字典类型数据\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 第三方应用需拥有“企业客户”权限。\n            - 第三方/自建应用只能获取到可见范围内的配置了客户联系功能的成员。\n\n        .. _获取客户列表: https://work.weixin.qq.com/api/doc/90000/90135/92113\n        '
        return self._get('externalcontact/list', params={'userid': userid})

    def batch_get_by_user(self, userid: str, cursor: str='', limit: int=50) -> dict:
        if False:
            print('Hello World!')
        '\n        批量获取客户详情\n\n        使用示例：\n\n        .. code-block:: python\n\n            from wechatpy.work import WeChatClient\n\n            # 需要注意使用正确的secret，否则会导致在之后的接口调用中失败\n            client = WeChatClient("corp_id", "secret_key")\n            # 批量获取该企业员工添加的客户(外部联系人)的详情\n            external_contact_list = client.external_contact.batch_get_by_user("user_id", "cursor", 10)["external_contact_list"]\n\n        :param userid: 企业成员的userid\n        :param cursor: 用于分页查询的游标，字符串类型，由上一次调用返回，首次调用可不填\n        :param limit: 返回的最大记录数，整型，最大值100，默认值50，超过最大值时取最大值\n        :return: 包含该企业员工添加的部分客户详情列表的字典类型数据\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 第三方应用需具有“企业客户权限->客户基础信息”权限\n            - 第三方/自建应用调用此接口时，userid需要在相关应用的可见范围内。\n\n        .. _批量获取客户详情: https://work.weixin.qq.com/api/doc/90000/90135/92994\n        '
        data = optionaldict(userid=userid, cursor=cursor, limit=limit)
        return self._post('externalcontact/batch/get_by_user', data=data)

    def gen_all_by_user(self, userid: str, limit: int=50) -> Iterator[dict]:
        if False:
            for i in range(10):
                print('nop')
        '\n        获取企业员工添加的所有客户详情列表的生成器\n\n        .. code-block:: python\n\n            from wechatpy.work import WeChatClient\n\n            # 需要注意使用正确的secret，否则会导致在之后的接口调用中失败\n            client = WeChatClient("corp_id", "secret_key")\n            #  获取企业员工添加的所有客户详情列表\n            for i in client.external_contact.gen_all_by_user("user_id", 10):\n                print(i)\n\n        :param userid: 企业员工userid\n        :param limit: 每次需要请求微信接口时返回的最大记录数，整型，最大值100，默认值50，超过最大值时取最大值\n        :return: 企业员工添加的所有客户详情列表的生成器\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 第三方应用需具有“企业客户权限->客户基础信息”权限\n            - 第三方/自建应用调用此接口时，userid需要在相关应用的可见范围内。\n        '
        cursor = ''
        while True:
            response = self.batch_get_by_user(userid, cursor, limit)
            if response.get('errcode') == 0:
                yield from response.get('external_contact_list', [])
            if response.get('next_cursor'):
                cursor = response['next_cursor']
            else:
                break

    def get(self, external_userid: str) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        获取客户详情\n\n        企业可通过此接口，根据 `外部联系人的userid（如何获取?）`_，拉取客户详情。\n\n        详细请查阅企业微信官方文档 `获取客户详情`_ 章节。\n\n        使用示例:\n\n        .. code-block:: python\n\n            from wechatpy.work import WeChatClient\n\n            # 需要注意使用正确的secret，否则会导致在之后的接口调用中失败\n            client = WeChatClient("corp_id", "secret_key")\n            # 接口数据\n            data = client.external_contact.get("external_userid")\n            # 外部联系人的自定义展示信息\n            external_profile = data["external_profile"]  # type: dict\n            # 部联系人的企业成员userid\n            follow_users = data["follow_user"]  # type: List[dict]\n\n        :param external_userid: 外部联系人的userid，注意不是企业成员的帐号\n        :return: 用户信息（字段内容请参考官方文档 `获取客户详情`_ 章节）\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 第三方/自建应用调用时，返回的跟进人follow_user仅包含应用可见范围之内的成员。\n\n        .. _外部联系人的userid（如何获取?）: https://work.weixin.qq.com/api/doc/9\n            0000/90135/92114#15445\n\n        .. _获取客户详情: https://work.weixin.qq.com/api/doc/90000/90135/92114\n        '
        return self._get('externalcontact/get', params={'external_userid': external_userid})

    def add_contact_way(self, type: int, scene: int, style: Optional[int]=None, remark: Optional[str]=None, skip_verify: bool=True, state: Optional[str]=None, user: List[str]=None, party: List[int]=None, is_temp: bool=False, expires_in: Optional[int]=None, chat_expires_in: Optional[int]=None, unionid: Optional[str]=None, conclusions: Optional[dict]=None) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        配置客户联系「联系我」方式\n\n        详细请查阅企业微信官方文档 `配置客户联系「联系我」方式`_ 章节。\n\n        **注意:**\n\n        - 每个联系方式最多配置100个使用成员（包含部门展开后的成员）\n        - 当设置为临时会话模式时（即 ``is_temp`` 为 `True` ），联系人仅支持配置为单人，\n          暂不支持多人\n        - 使用 ``unionid`` 需要调用方（企业或服务商）的企业微信“客户联系”中已绑定微信开\n          发者账户\n\n        使用示例:\n\n        .. code-block:: python\n\n            from wechatpy.work import WeChatClient\n\n            # 需要注意使用正确的secret，否则会导致在之后的接口调用中失败\n            client = WeChatClient("corp_id", "secret_key")\n            # 调用接口\n            result = client.external_contact.add_contact_way(\n                type=1,\n                scene=1,\n                style=1,\n                remark="渠道用户",\n                skip_verify=True,\n                state="teststate",\n                user=["zhansan", "lisi"],\n                party=[2,3],\n                is_temp=True,\n                expires_in=86400,\n                chat_expires_in=86400,\n                unionid="oxTWIuGaIt6gTKsQRLau2M0AAAA",\n                conclusions={\n                    "text": {"content": "文本消息内容"},\n                    "image": {"media_id": "MEDIA_ID"},\n                    "link": {\n                        "title": "消息标题",\n                        "picurl": "https://example.pic.com/path",\n                        "desc": "消息描述",\n                        "url": "https://example.link.com/path",\n                    },\n                    "miniprogram": {\n                        "title": "消息标题",\n                        "pic_media_id": "MEDIA_ID",\n                        "appid": "wx8bd80126147dfAAA",\n                        "page": "/path/index.html",\n                    },\n                },\n            )\n            # 新增联系方式的配置id\n            config_id = result["config_id"]\n            # 联系我二维码链接，仅在scene为2时返回\n            qr_code = result.get("qr_code")\n\n        :param type: 联系方式类型,1-单人, 2-多人\n        :param scene: 场景，1-在小程序中联系，2-通过二维码联系\n        :param style: 在小程序中联系时使用的控件样式，详见附表\n        :param remark: 联系方式的备注信息，用于助记，不超过30个字符\n        :param skip_verify: 外部客户添加时是否无需验证，默认为true\n        :param state: 企业自定义的state参数，用于区分不同的添加渠道，在调用\n            `获取外部联系人详情`_ 时会返回该参数值\n        :param user: 使用该联系方式的用户userID列表，在type为1时为必填，且只能有一个\n        :param party: 使用该联系方式的部门id列表，只在type为2时有效\n        :param is_temp: 是否临时会话模式，``True`` 表示使用临时会话模式，默认为 ``False``\n        :param expires_in: 临时会话二维码有效期，以秒为单位。该参数仅在 ``is_temp`` 为\n            ``True`` 时有效，默认7天\n        :param chat_expires_in: 临时会话有效期，以秒为单位。该参数仅在 ``is_temp`` 为\n            ``True`` 时有效，默认为添加好友后24小时\n        :param unionid: 可进行临时会话的客户unionid，该参数仅在 ``is_temp`` 为\n            ``True``时有效，如不指定则不进行限制\n        :param conclusions: 结束语，会话结束时自动发送给客户，可参考 `结束语定义`_，\n            仅在 ``is_temp`` 为 ``True`` 时有效\n        :return: 返回的 JSON 数据包\n\n        .. note::\n            **调用接口应满足如下的权限要求：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 使用人员需要配置了 `客户联系功能`_。\n            - 第三方调用时，应用需具有 `企业客户权限`_。\n            - 第三方/自建应用调用时，传入的userid和partyid需要在此应用的可见范围内。\n            - 配置的使用成员必须在企业微信激活且已经过实名认证。\n            - 临时会话的二维码具有有效期，添加企业成员后仅能在指定有效期内进行会话，\n              仅支持医疗行业企业创建。\n              临时会话模式可以配置会话结束时自动发送给用户的结束语。\n\n        .. _配置客户联系「联系我」方式: https://work.weixin.qq.com/api/doc/90000/9\n            0135/92572#%E9%85%8D%E7%BD%AE%E5%AE%A2%E6%88%B7%E8%81%94%E7%B3%BB\n            %E3%80%8C%E8%81%94%E7%B3%BB%E6%88%91%E3%80%8D%E6%96%B9%E5%BC%8F\n\n        .. _结束语定义: https://work.weixin.qq.com/api/doc/90000/90135/92572#156\n            45/%E7%BB%93%E6%9D%9F%E8%AF%AD%E5%AE%9A%E4%B9%89\n        '
        data = optionaldict(type=type, scene=scene, style=style, remark=remark, skip_verify=skip_verify, state=state, user=user, party=party, is_temp=is_temp, expires_in=expires_in, chat_expires_in=chat_expires_in, unionid=unionid, conclusions=conclusions)
        return self._post('externalcontact/add_contact_way', data=data)

    def get_contact_way(self, config_id: str) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        获取企业已配置的「联系我」方式\n\n        批量获取企业配置的「联系我」二维码和「联系我」小程序按钮。\n\n        详细请查阅企业微信官方文档 `获取企业已配置的「联系我」方式`_ 章节。\n\n        :param config_id: 联系方式的配置id, e.g.42b34949e138eb6e027c123cba77fad7\n        :return: 返回的 JSON 数据包\n\n        .. note::\n            **调用接口应满足如下的权限要求：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 使用人员需要配置了 `客户联系功能`_。\n            - 第三方调用时，应用需具有 `企业客户权限`_。\n            - 第三方/自建应用调用时，传入的userid和partyid需要在此应用的可见范围内。\n            - 配置的使用成员必须在企业微信激活且已经过实名认证。\n            - 临时会话的二维码具有有效期，添加企业成员后仅能在指定有效期内进行会话，\n              仅支持医疗行业企业创建。\n              临时会话模式可以配置会话结束时自动发送给用户的结束语。\n\n        .. _获取企业已配置的「联系我」方式: https://work.weixin.qq.com/api/doc/90000\n           /90135/92572#%E8%8E%B7%E5%8F%96%E4%BC%81%E4%B8%9A%E5%B7%B2%E9%85%8D\n           %E7%BD%AE%E7%9A%84%E3%80%8C%E8%81%94%E7%B3%BB%E6%88%91%E3%80%8D%E6%\n           96%B9%E5%BC%8F\n        '
        data = optionaldict(config_id=config_id)
        return self._post('externalcontact/get_contact_way', data=data)

    def update_contact_way(self, config_id, remark, skip_verify=True, style=None, state=None, user=None, party=None) -> dict:
        if False:
            return 10
        '\n        更新企业已配置的「联系我」方式\n\n        更新企业配置的「联系我」二维码和「联系我」小程序按钮中的信息，如使用人员和备注等。\n\n        详细请查阅企业微信官方文档 `更新企业已配置的「联系我」方式`_ 章节。\n\n        :param config_id: 企业联系方式的配置id\n        :param remark: 联系方式的备注信息，不超过30个字符，将覆盖之前的备注\n        :param skip_verify: 外部客户添加时是否无需验证\n        :param style: 样式，只针对“在小程序中联系”的配置生效\n        :param state: 企业自定义的state参数，用于区分不同的添加渠道，在调用“获取外部联系\n           人详情”时会返回该参数值\n        :param user: 使用该联系方式的用户列表，将覆盖原有用户列表\n        :param party: 使用该联系方式的部门列表，将覆盖原有部门列表，只在配置的type为2时有效\n        :return: 返回的 JSON 数据包\n\n        .. note::\n            **调用接口应满足如下的权限要求：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 使用人员需要配置了 `客户联系功能`_。\n            - 第三方调用时，应用需具有 `企业客户权限`_。\n            - 第三方/自建应用调用时，传入的userid和partyid需要在此应用的可见范围内。\n            - 配置的使用成员必须在企业微信激活且已经过实名认证。\n            - 临时会话的二维码具有有效期，添加企业成员后仅能在指定有效期内进行会话，\n              仅支持医疗行业企业创建。\n              临时会话模式可以配置会话结束时自动发送给用户的结束语。\n\n        .. _更新企业已配置的「联系我」方式: https://work.weixin.qq.com/api/doc/90000\n           /90135/92572#%E6%9B%B4%E6%96%B0%E4%BC%81%E4%B8%9A%E5%B7%B2%E9%85%8D\n           %E7%BD%AE%E7%9A%84%E3%80%8C%E8%81%94%E7%B3%BB%E6%88%91%E3%80%8D%E6%\n           96%B9%E5%BC%8F\n        '
        data = optionaldict(config_id=config_id, remark=remark, skip_verify=skip_verify, style=style, state=state, user=user, party=party)
        return self._post('externalcontact/update_contact_way', data=data)

    def del_contact_way(self, config_id: str) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        删除企业已配置的「联系我」方式\n\n        删除一个已配置的「联系我」二维码或者「联系我」小程序按钮。\n\n        详细请查阅企业微信官方文档 `删除企业已配置的「联系我」方式`_ 章节。\n\n        :param config_id: 企业联系方式的配置id\n        :return: 返回的 JSON 数据包\n\n        .. note::\n            **调用接口应满足如下的权限要求：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 使用人员需要配置了 `客户联系功能`_。\n            - 第三方调用时，应用需具有 `企业客户权限`_。\n            - 第三方/自建应用调用时，传入的userid和partyid需要在此应用的可见范围内。\n            - 配置的使用成员必须在企业微信激活且已经过实名认证。\n            - 临时会话的二维码具有有效期，添加企业成员后仅能在指定有效期内进行会话，\n              仅支持医疗行业企业创建。\n              临时会话模式可以配置会话结束时自动发送给用户的结束语。\n\n        .. _删除企业已配置的「联系我」方式: https://work.weixin.qq.com/api/doc/90000\n           /90135/92572#%E5%88%A0%E9%99%A4%E4%BC%81%E4%B8%9A%E5%B7%B2%E9%85%8D\n           %E7%BD%AE%E7%9A%84%E3%80%8C%E8%81%94%E7%B3%BB%E6%88%91%E3%80%8D%E6%\n           96%B9%E5%BC%8F\n\n        '
        data = optionaldict(config_id=config_id)
        return self._post('externalcontact/del_contact_way', data=data)

    def add_msg_template(self, template: dict) -> dict:
        if False:
            print('Hello World!')
        '\n        添加企业群发消息模板\n\n        企业可通过此接口添加企业群发消息的任务并通知客服人员发送给相关客户或客户群。\n        （注：企业微信终端需升级到2.7.5版本及以上）\n\n        **注意**：调用该接口并不会直接发送消息给客户/客户群，需要相关的客服人员操作以后才会\n        实际发送（客服人员的企业微信需要升级到2.7.5及以上版本）\n\n        同一个企业每个自然月内仅可针对一个客户/客户群发送4条消息，超过限制的用户将会被忽略。\n\n        详细请查阅企业微信官方文档 `添加企业群发消息任务`_ 章节。\n\n        使用示例:\n\n        .. code-block:: python\n\n            from wechatpy.exceptions import WeChatClientException\n            from wechatpy.work import WeChatClient\n\n            # 需要注意使用正确的secret，否则会导致在之后的接口调用中失败\n            client = WeChatClient("corp_id", "secret_key")\n            template = {\n                "chat_type":"single",\n                "external_userid":[\n                    "woAJ2GCAAAXtWyujaWJHDDGi0mACAAAA",\n                    "wmqfasd1e1927831123109rBAAAA"\n                ],\n                "sender":"zhangsan",\n                "text":{\n                    "content":"文本消息内容"\n                },\n                "image":{\n                    "media_id":"MEDIA_ID",\n                    "pic_url":"http://p.qpic.cn/pic_wework/3474110808/7a6344sdadfwehe42060/0"\n                },\n                "link":{\n                    "title":"消息标题",\n                    "picurl":"https://example.pic.com/path",\n                    "desc":"消息描述",\n                    "url":"https://example.link.com/path"\n                },\n                "miniprogram":{\n                    "title":"消息标题",\n                    "pic_media_id":"MEDIA_ID",\n                    "appid":"wx8bd80126147dfAAA",\n                    "page":"/path/index.html"\n                }\n            }\n            try:\n                result = client.external_contact.add_msg_template(template=template)\n                # 无效或无法发送的external_userid列表\n                fail_list = result["fail_list"]\n                # 企业群发消息的id，可用于获取群发消息发送结果\n                msgid = result["msgid]\n            except WeChatClientException as err:\n                # 接口调用失败\n                ...\n\n        :param template: 参考官方文档和使用示例\n        :return: 请求结果（字典类型）\n\n        .. _添加企业群发消息任务: https://work.weixin.qq.com/api/doc/90000/90135/92135\n        '
        return self._post('externalcontact/add_msg_template', data=template)

    def get_group_msg_result(self, msgid):
        if False:
            i = 10
            return i + 15
        '\n        获取企业群发消息发送结果\n\n        企业和第三方可通过该接口获取到添加企业群发消息模板生成消息的群发发送结果。\n        https://work.weixin.qq.com/api/doc#90000/90135/91561\n\n        :param msgid: 群发消息的id，通过添加企业群发消息模板接口返回\n        :return: 返回的 JSON 数据包\n        '
        data = optionaldict(msgid=msgid)
        return self._post('externalcontact/get_group_msg_result', data=data)

    def get_user_behavior_data(self, userid: Optional[List[str]], start_time: int, end_time: int, partyid: Optional[List[str]]=None) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        获取「联系客户统计」数据\n\n        企业可通过此接口获取成员联系客户的数据，包括发起申请数、新增客户数、聊天数、发送消息\n        数和删除/拉黑成员的客户数等指标。\n\n        详细请查阅企业微信官方文档 `获取「联系客户统计」数据`_ 章节。\n\n        :param userid: userid列表\n        :param partyid: 部门ID列表，最多100个\n        :param start_time: 数据起始时间\n        :param end_time: 数据结束时间\n        :return: 返回的 JSON 数据包\n        :raises AssertionError: 当userid和partyid同时为空时抛出该移除\n\n        .. warning::\n\n           1. ``userid`` 和 ``partyid`` 不可同时为空;\n           2. 此接口提供的数据以天为维度，查询的时间范围为 ``[START_TIME,END_TIME]``，\n              即前后均为闭区间，支持的最大查询跨度为30天；\n           3. 用户最多可获取最近180天内的数据；\n           4. 当传入的时间不为0点时间戳时，会向下取整，如传入\n              1554296400(wED aPR 3 21:00:00 cst 2019) 会被自动转换为\n              1554220800（wED aPR 3 00:00:00 cst 2019）;\n           5. 如传入多个 ``USERID``，则表示获取这些成员总体的联系客户数据。\n\n        .. note::\n\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 第三方应用使用，需具有“企业客户权限->客户联系->获取成员联系客户的数据统计”权限。\n            - 第三方/自建应用调用时传入的userid和partyid要在应用的可见范围内;\n\n        .. _获取「联系客户统计」数据: https://work.weixin.qq.com/api/doc/90000/90135/92132\n        '
        assert userid or partyid, 'userid和partyid不可同时为空'
        data = optionaldict(userid=userid, start_time=start_time, end_time=end_time, partyid=partyid)
        return self._post('externalcontact/get_user_behavior_data', data=data)

    def send_welcome_msg(self, template: dict) -> dict:
        if False:
            return 10
        '\n        发送新客户欢迎语\n\n        企业微信在向企业推送 `添加外部联系人事件`_ 时，会额外返回一个welcome_code，企业以\n        此为凭据调用接口，即可通过成员向新添加的客户发送个性化的欢迎语。\n\n        为了保证用户体验以及避免滥用，企业仅可在收到相关事件后20秒内调用，且只可调用一次。如\n        果企业已经在管理端为相关成员配置了可用的欢迎语，则推送添加外部联系人事件时不会返回\n        welcome_code。\n\n        每次添加新客户时 **可能有多个企业自建应用/第三方应用收到带有welcome_code的回调事件**，\n        但仅有最先调用的可以发送成功。后续调用将返回 **41051（externaluser has started\n        chatting）** 错误，请用户根据实际使用需求，合理设置应用可见范围，避免冲突。\n\n        详细请查阅企业微信官方文档 `发送新客户欢迎语`_ 章节。\n\n        使用示例:\n\n        .. code-block:: python\n\n            from wechatpy.exceptions import WeChatClientException\n            from wechatpy.work import WeChatClient\n\n            # 需要注意使用正确的secret，否则会导致在之后的接口调用中失败\n            client = WeChatClient("corp_id", "secret_key")\n            template = {\n                "chat_type":"single",\n                "external_userid":[\n                    "woAJ2GCAAAXtWyujaWJHDDGi0mACAAAA",\n                    "wmqfasd1e1927831123109rBAAAA"\n                ],\n                "sender":"zhangsan",\n                "text":{\n                    "content":"文本消息内容"\n                },\n                "image":{\n                    "media_id":"MEDIA_ID",\n                    "pic_url":"http://p.qpic.cn/pic_wework/3474110808/7a6344sdadfwehe42060/0"\n                },\n                "link":{\n                    "title":"消息标题",\n                    "picurl":"https://example.pic.com/path",\n                    "desc":"消息描述",\n                    "url":"https://example.link.com/path"\n                },\n                "miniprogram":{\n                    "title":"消息标题",\n                    "pic_media_id":"MEDIA_ID",\n                    "appid":"wx8bd80126147dfAAA",\n                    "page":"/path/index.html"\n                }\n            }\n            try:\n                client.external_contact.send_welcome_msg(template=template)\n            except WeChatClientException as err:\n                # 消息发送失败时的处理\n                ...\n\n        :param template: 参考官方文档和使用示例\n        :return: 消息推送结果（字典类型）\n\n        .. _添加外部联系人事件: https://work.weixin.qq.com/api/doc/90000/90135/921\n            37#15260/%E6%B7%BB%E5%8A%A0%E5%A4%96%E9%83%A8%E8%81%94%E7%B3%BB%E4%\n            BA%BA%E4%BA%8B%E4%BB%B6\n\n        .. _发送新客户欢迎语: https://work.weixin.qq.com/api/doc/90000/90135/92137\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 第三方应用需要拥有「企业客户」权限，且企业成员处于相关应用的可见范围内。\n        '
        return self._post('externalcontact/send_welcome_msg', data=template)

    def get_unassigned_list(self, page_id: int=0, page_size: int=1000, cursor: Optional[str]=None) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        获取离职成员列表\n\n        企业和第三方可通过此接口，获取所有离职成员的客户列表，并可进一步调用\n        `分配在职或离职成员的客户`_ 接口将这些客户重新分配给其他企业成员。\n\n        详细请查阅企业微信官方文档 `获取离职成员列表`_ 章节。\n\n        :param page_id: 分页查询，要查询页号，从0开始\n        :param page_size: 每次返回的最大记录数，默认为1000，最大值为1000\n        :param cursor: 分页查询游标，字符串类型，适用于数据量较大的情况，如果使用该参数\n           则无需填写page_id，该参数由上一次调用返回\n        :return: 响应结果\n\n        .. note::\n           当 ``page_id`` 为1，``page_size`` 为100时，表示取第101到第200条记录。\n           由于每个成员的客户数不超过5万，故 ``page_id * page_size`` 必须小于5万。\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 第三方应用需拥有“企业客户权限->客户联系->分配在职或离职成员的客户”权限\n\n        .. _获取离职成员列表: https://work.weixin.qq.com/api/doc/90000/90135/92124\n        '
        data = optionaldict(page_id=page_id, page_size=page_size, cursor=cursor)
        return self._post('externalcontact/get_unassigned_list', data=data)

    def transfer(self, external_userid: str, handover_userid: str, takeover_userid: str, transfer_success_msg: Optional[str]=None) -> dict:
        if False:
            while True:
                i = 10
        '\n        分配在职或离职成员的客户\n\n\n        企业可通过此接口，转接在职成员的客户或分配离职成员的客户给其他成员。\n\n        详细请查阅企业微信官方文档 `分配在职或离职成员的客户`_ 章节。\n\n        **调用参数注意事项：**\n\n        - 在某些特殊情景下，可能存在已离职的成员和当前在职的企业成员具有相同userid的情况，\n          此时优先分配在职成员的客户.\n        - ``external_userid`` 必须是 ``handover_userid`` 的客户\n          （即 `配置了客户联系功能`_ 的成员所添加的联系人）。\n        - 在职成员的每位客户最多被分配2次。客户被转接成功后，将有90个自然日的服务关系保护期，\n          保护期内的客户无法再次被分配。\n\n        使用示例:\n\n        .. code-block:: python\n\n            from wechatpy.exceptions import WeChatClientException\n            from wechatpy.work import WeChatClient\n\n            # 需要注意使用正确的secret，否则会导致在之后的接口调用中失败\n            client = WeChatClient("corp_id", "secret_key")\n            try:\n                client.external_contact.transfer(\n                    external_userid="woAJ2GCAAAXtWyujaWJHDDGi0mACAAAA",\n                    handover_userid="zhangsan",\n                    takeover_userid="lisi",\n                    transfer_success_msg="您好！",您好\n                )\n            except WeChatClientException as err:\n                # 分配失败时的处理\n                ...\n\n        :param external_userid: 外部联系人的userid，注意不是企业成员的帐号\n        :param handover_userid: 离职成员的userid\n        :param takeover_userid: 接替成员的userid\n        :param transfer_success_msg: 转移成功后发给客户的消息，最多200个字符，不填则\n            使用默认文案，目前只对在职成员分配客户的情况生效\n        :return: 分配结果（字典类型）\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 第三方应用需拥有 `企业客户权限`_。\n            - 接替成员必须在此第三方应用或自建应用的可见范围内。\n            - 接替成员需要配置了 `客户联系功能`_。\n            - 接替成员需要在企业微信激活且已经过实名认证。\n\n        .. _分配在职或离职成员的客户: https://work.weixin.qq.com/api/doc/90000/90135/92125\n\n        .. _配置了客户联系功能: https://work.weixin.qq.com/api/doc/90000/90135/\n            92125#13473/%E5%BC%80%E5%A7%8B%E5%BC%80%E5%8F%91\n\n\n        '
        data = optionaldict(external_userid=external_userid, handover_userid=handover_userid, takeover_userid=takeover_userid, transfer_success_msg=transfer_success_msg)
        return self._post('externalcontact/transfer', data=data)

    def get_corp_tag_list(self, tag_ids: Optional[List[str]]=None) -> dict:
        if False:
            print('Hello World!')
        '\n        获取企业标签库\n\n        企业可通过此接口获取企业客户标签详情。\n\n        **企业客户标签** 是针对企业的外部联系人进行标记和分类的标签，由企业统一配置后，企业\n        成员可使用此标签对客户进行标记。\n\n        详细请查阅企业微信官方文档 `获取企业标签库`_ 章节。\n\n        使用示例:\n\n        .. code-block:: python\n\n            from wechatpy.work import WeChatClient\n\n            # 需要注意使用正确的secret，否则会导致在之后的接口调用中失败\n            client = WeChatClient("corp_id", "secret_key")\n            # 接口数据\n            data = client.external_contact.get_corp_tag_list(["tag_id1", "tag_id2"])\n            # 标签组列表\n            tag_groups = data["tag_group"]  # type: List[dict]\n\n        :param tag_ids: 需要查询的标签id，如果为 ``None`` 则获取该企业的所有客户标签，\n            目前暂不支持标签组id。\n        :return: 包含标签信息的字典类型数据（详细字段请参考 `获取企业标签库`_）\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 对于添加/删除/编辑企业客户标签接口，目前仅支持使用 `客户联系secret`_ 所获\n              取的accesstoken来调用。\n            - 第三方仅可读取，且应用需具有 `企业客户权限`_。\n\n        .. _获取企业标签库: https://work.weixin.qq.com/api/doc/90000/90135/92117\n            #%E8%8E%B7%E5%8F%96%E4%BC%81%E4%B8%9A%E6%A0%87%E7%AD%BE%E5%BA%93\n        '
        data = optionaldict(tag_id=tag_ids)
        return self._post('externalcontact/get_corp_tag_list', data=data)

    def add_corp_tag(self, group_id: Optional[str], group_name: Optional[str], order: Optional[int], tags: dict) -> dict:
        if False:
            print('Hello World!')
        '\n        添加企业客户标签\n\n        **企业客户标签** 是针对企业的外部联系人进行标记和分类的标签，由企业统一配置后，企业\n        成员可使用此标签对客户进行标记。\n\n        企业可通过此接口向客户标签库中添加新的标签组和标签，**每个企业最多可配置3000个企业标签**。\n\n        详细请查阅企业微信官方文档 `添加企业客户标签`_ 章节。\n\n        **参数注意事项:**\n\n        - 如果要向指定的标签组下添加标签，需要提供 ``group_id`` 参数；如果要创建一个全新的\n          标签组以及标签，则需要通过 ``group_name`` 参数指定新标签组名称，如果填写的\n          ``group_name`` 已经存在，则会在此标签组下新建标签。\n        - 如果提供了 ``group_id`` 参数，则 ``group_name`` 和标签组的 ``order`` 参数\n          会被忽略。\n        - 不支持创建空标签组。\n        - 标签组内的标签不可同名，如果传入多个同名标签，则只会创建一个。\n\n        使用示例:\n\n        .. code-block:: python\n\n            from wechatpy.work import WeChatClient\n\n            # 需要注意使用正确的secret，否则会导致在之后的接口调用中失败\n            client = WeChatClient("corp_id", "secret_key")\n            # 创建标签\n            result = client.external_contact.add_corp_tag(\n                group_id="GROUP_ID",\n                group_name="GROUP_NAME",\n                order=1,\n                tags=[\n                    {"name": "TAG_NAME_1", "order": 1},\n                    {"name": "TAG_NAME_2", "order": 2},\n                ],\n            )\n            # 创建成功后的标签组信息\n            tag_group = result["tag_group"]  # type: dict\n\n        :param group_id: 标签组id\n        :param group_name: 标签组名称，最长为30个字符\n        :param order: 标签组次序值。order值大的排序靠前。有效的值范围是[0, 2^32)\n        :param tags: 需要添加的标签列表，标签信息是包含 ``name`` （必须）和 ``order``\n            （可选）两个字段的字典类型数据，比如: ``[{"name": \'tag_name", "order": 1}]``\n            ，或者 ``[{"name": "tag_name"}]``。\n        :return: 标签创建结果（字典类型，字段请参考 `添加企业客户标签`_）\n\n        .. _添加企业客户标签: https://work.weixin.qq.com/api/doc/90000/90135/921\n            17#%E6%B7%BB%E5%8A%A0%E4%BC%81%E4%B8%9A%E5%AE%A2%E6%88%B7%E6%A0%8\n            7%E7%AD%BE\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 对于添加/删除/编辑企业客户标签接口，目前仅支持使用 `客户联系secret`_ 所获\n              取的accesstoken来调用。\n            - 第三方仅可读取，且应用需具有 `企业客户权限`_。\n\n        .. warning:: 暂不支持第三方调用。\n        '
        data = optionaldict(group_id=group_id, group_name=group_name, order=order, tag=tags)
        return self._post('externalcontact/add_corp_tag', data=data)

    def edit_corp_tag(self, id: str, name: Optional[str]=None, order: Optional[int]=None) -> dict:
        if False:
            while True:
                i = 10
        '\n        编辑企业客户标签\n\n        企业可通过此接口编辑客户标签/标签组的名称或次序值。\n\n        **注意**: 修改后的标签组不能和已有的标签组重名，标签也不能和同一标签组下的其他标签重名。\n\n        详细请查阅企业微信官方文档 `编辑企业客户标签`_ 章节。\n\n        使用示例:\n\n        .. code-block:: python\n\n            from wechatpy.exceptions import WeChatClientException\n            from wechatpy.work import WeChatClient\n\n            # 需要注意使用正确的secret，否则会导致在之后的接口调用中失败\n            client = WeChatClient("corp_id", "secret_key")\n            # 修改标签\n            try:\n                client.external_contact.edit_corp_tag(id="TAG_ID", name="NEW_TAG_NAME")\n            except WeChatClientException as err:\n                # 标签修改失败时的处理\n                ...\n\n        :param id: 标签或标签组的id列表\n        :param name: 新的标签或标签组名称，最长为30个字符\n        :param order: 标签/标签组的次序值。order值大的排序靠前。有效的值范围是[0, 2^32)\n        :return: 创建结果（字典类型）\n\n        .. _编辑企业客户标签: https://work.weixin.qq.com/api/doc/90000/90135/9211\n            7#%E7%BC%96%E8%BE%91%E4%BC%81%E4%B8%9A%E5%AE%A2%E6%88%B7%E6%A0%87%\n            E7%AD%BE\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 对于添加/删除/编辑企业客户标签接口，目前仅支持使用 `客户联系secret`_ 所获\n              取的accesstoken来调用。\n            - 第三方仅可读取，且应用需具有 `企业客户权限`_。\n\n        .. warning:: 暂不支持第三方调用。\n        '
        data = optionaldict(id=id, name=name, order=order)
        return self._post('externalcontact/edit_corp_tag', data=data)

    def del_corp_tag(self, tag_id: Optional[str]=None, group_id: Optional[str]=None) -> dict:
        if False:
            print('Hello World!')
        '\n        删除企业客户标签\n\n        企业可通过此接口删除客户标签库中的标签，或删除整个标签组。\n\n        详细请查阅企业微信官方文档 `删除企业客户标签`_ 章节。\n\n        **参数注意事项**:\n\n        - ``tag_id`` 和 ``group_id`` 不可同时为空。\n        - 如果一个标签组下所有的标签均被删除，则标签组会被自动删除。\n\n        使用示例:\n\n        .. code-block:: python\n\n            from wechatpy.exceptions import WeChatClientException\n            from wechatpy.work import WeChatClient\n\n            # 需要注意使用正确的secret，否则会导致在之后的接口调用中失败\n            client = WeChatClient("corp_id", "secret_key")\n            try:\n                client.external_contact.del_corp_tag(tag_id="TAG_ID")\n            except WeChatClientException as err:\n                # 标签删除失败时的处理\n                ...\n\n        :param tag_id: 标签的id列表\n        :param group_id: 标签组的id列表\n        :return: 删除结果（字典类型）\n\n        .. _删除企业客户标签: https://work.weixin.qq.com/api/doc/90000/90135/9211\n            7#%E5%88%A0%E9%99%A4%E4%BC%81%E4%B8%9A%E5%AE%A2%E6%88%B7%E6%A0%87%\n            E7%AD%BE\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 对于添加/删除/编辑企业客户标签接口，目前仅支持使用 `客户联系secret`_ 所获\n              取的accesstoken来调用。\n            - 第三方仅可读取，且应用需具有 `企业客户权限`_。\n\n        .. warning:: 暂不支持第三方调用。\n        '
        data = optionaldict(tag_id=tag_id, group_id=group_id)
        return self._post('externalcontact/del_corp_tag', data=data)

    def mark_tag(self, userid: str, external_userid: str, add_tag: Optional[List[str]]=None, remove_tag: Optional[List[str]]=None) -> dict:
        if False:
            print('Hello World!')
        '\n        编辑客户企业标签\n\n        企业可通过此接口为指定成员的客户添加上由 `企业统一配置的标签`_。\n\n        详细请查阅企业微信官方文档 `编辑客户企业标签`_ 章节。\n\n        **参数注意事项**:\n\n        - 请确保 ``external_userid`` 是 ``userid`` 的外部联系人。\n        - ``add_tag`` 和 ``remove_tag`` 不可同时为空。\n        - 同一个标签组下现已支持多个标签\n\n        使用示例:\n\n        .. code-block:: python\n\n            from wechatpy.exceptions import WeChatClientException\n            from wechatpy.work import WeChatClient\n\n            # 需要注意使用正确的secret，否则会导致在之后的接口调用中失败\n            client = WeChatClient("corp_id", "secret_key")\n            try:\n                client.external_contact.mark_tag(\n                    userid="USER_ID",\n                    external_userid="EXT_ID",\n                    add_tag=["TAG_ID_1", "TAG_ID_2"],\n                )\n            except WeChatClientException as err:\n                # 编辑失败时的处理\n                ...\n\n        :param userid: 添加外部联系人的userid\n        :param external_userid: 外部联系人userid\n        :param add_tag: 要标记的标签列表\n        :param remove_tag: 要移除的标签列表\n        :return: 处理结果（字典类型）\n\n        .. _企业统一配置的标签: https://work.weixin.qq.com/api/doc/90000/90135/92\n            118#17298\n\n        .. _编辑客户企业标签: https://work.weixin.qq.com/api/doc/90000/90135/92118\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 第三方调用时，应用需具有外部联系人管理权限。\n        '
        add_tag = add_tag or []
        remove_tag = remove_tag or []
        data = optionaldict(userid=userid, external_userid=external_userid, add_tag=add_tag, remove_tag=remove_tag)
        return self._post('externalcontact/mark_tag', data=data)

    def get_group_chat_list(self, limit: int, status_filter: int=0, owner_filter: Optional[dict]=None, cursor: Optional[str]=None) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        获取客户群列表\n\n        该接口用于获取配置过客户群管理的客户群列表。\n\n        详细请查阅企业微信官方文档 `获取客户群列表`_ 章节。\n\n        :param limit: 分页，预期请求的数据量，取值范围 1 ~ 1000\n        :param status_filter: 客户群跟进状态过滤（默认为0）。\n            0: 所有列表(即不过滤)\n            1: 离职待继承\n            2: 离职继承中\n            3: 离职继承完成\n        :param owner_filter: 群主过滤。\n            如果不填，表示获取应用可见范围内全部群主的数据\n            （但是不建议这么用，如果可见范围人数超过1000人，为了防止数据包过大，会报错 81017）\n        :param cursor: 用于分页查询的游标，字符串类型，由上一次调用返回，首次调用不填\n        :return: 响应数据\n\n        .. warning::\n\n           如果不指定 ``owner_filter``，会拉取应用可见范围内的所有群主的数据，\n           但是不建议这样使用。如果可见范围内人数超过1000人，为了防止数据包过大，\n           会报错 81017。此时，调用方需通过指定 ``owner_filter`` 来缩小拉取范围。\n\n           旧版接口以 ``offset+limit`` 分页，要求 ``offset+limit`` 不能超过50000，\n           该方案将废弃，请改用 ``cursor+limit`` 分页。\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 第三方应用需具有“企业客户权限->客户基础信息”权限\n            - 对于第三方/自建应用，群主必须在应用的可见范围。\n\n        .. _获取客户群列表: https://work.weixin.qq.com/api/doc/90000/90135/92120\n        '
        data = optionaldict(status_filter=status_filter, owner_filter=owner_filter, cursor=cursor, limit=limit)
        return self._post('externalcontact/groupchat/list', data=data)

    def get_group_chat_info(self, chat_id: str) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        获取客户群详情\n\n        通过客户群ID，获取详情。包括群名、群成员列表、群成员入群时间、入群方式。\n        （客户群是由具有客户群使用权限的成员创建的外部群）\n\n        需注意的是，如果发生群信息变动，会立即收到群变更事件，但是部分信息是异步处理，\n        可能需要等一段时间调此接口才能得到最新结果\n\n        详细请查阅企业微信官方文档 `获取客户群详情`_ 章节。\n\n        :param chat_id: 客户群ID\n        :return: 响应数据\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 第三方应用需具有“企业客户权限->客户基础信息”权限\n            - 对于第三方/自建应用，群主必须在应用的可见范围。\n\n        .. _获取客户群详情: https://work.weixin.qq.com/api/doc/90000/90135/92122\n        '
        data = optionaldict(chat_id=chat_id)
        return self._post('externalcontact/groupchat/get', data=data)

    def add_group_welcome_template(self, template: dict, agentid: Optional[int]=None) -> dict:
        if False:
            print('Hello World!')
        '\n        添加群欢迎语素材\n\n        企业可通过此API向企业的入群欢迎语素材库中添加素材。每个企业的入群欢迎语素材库中，\n        最多容纳100个素材。\n\n        详细请查阅企业微信官方文档 `添加群欢迎语素材`_ 章节。\n\n        :param template: 群欢迎语素材内容，详细字段请参考微信文档\n        :param agentid: 授权方安装的应用agentid。仅旧的第三方多应用套件需要填此参数\n        :return: 响应数据\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 第三方应用需具有“企业客户权限->客户联系->配置入群欢迎语素材”权限\n\n        .. _添加群欢迎语素材: https://work.weixin.qq.com/api/doc/90000/90135/\n            92366#%E6%B7%BB%E5%8A%A0%E5%85%A5%E7%BE%A4%E6%AC%A2%E8%BF%8E%E\n            8%AF%AD%E7%B4%A0%E6%9D%90\n\n        '
        data = optionaldict()
        data.update(template)
        data['agentid'] = agentid
        return self._post('externalcontact/group_welcome_template/add', data=data)

    def update_group_welcome_template(self, template: dict, template_id: str, agentid: Optional[int]=None) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        编辑群欢迎语素材\n\n        企业可通过此API编辑入群欢迎语素材库中的素材，且仅能够编辑调用方自己创建的入群欢迎语素材。\n\n\n        详细请查阅企业微信官方文档 `编辑群欢迎语素材`_ 章节。\n\n        :param template: 群欢迎语素材内容，详细字段请参考微信文档\n        :param template_id: 欢迎语素材id\n        :param agentid: 授权方安装的应用agentid。仅旧的第三方多应用套件需要填此参数\n        :return: 响应数据\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 第三方应用需具有“企业客户权限->客户联系->配置入群欢迎语素材”权限\n            - 仅可编辑本应用创建的入群欢迎语素材\n\n        .. _添编辑群欢迎语素材: https://work.weixin.qq.com/api/doc/90000/90135/\n            92366#%E7%BC%96%E8%BE%91%E5%85%A5%E7%BE%A4%E6%AC%A2%E8%BF%8E%E8%\n            AF%AD%E7%B4%A0%E6%9D%90\n        '
        data = optionaldict()
        data.update(template)
        data['template_id'] = template_id
        data['agentid'] = agentid
        return self._post('externalcontact/group_welcome_template/edit', data=data)

    def get_group_welcome_template(self, template_id: str) -> dict:
        if False:
            print('Hello World!')
        '\n        获取入群欢迎语素材\n\n        企业可通过此API获取入群欢迎语素材。\n\n        详细请查阅企业微信官方文档 `获取入群欢迎语素`_ 章节。\n\n        :param template_id: 群欢迎语的素材id\n        :return: 响应数据\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 第三方应用需具有“企业客户权限->客户联系->配置入群欢迎语素材”权限\n\n        .. _获取入群欢迎语素材: https://work.weixin.qq.com/api/doc/90000/90135/\n            92366#%E8%8E%B7%E5%8F%96%E5%85%A5%E7%BE%A4%E6%AC%A2%E8%BF%8E%E8%\n            AF%AD%E7%B4%A0%E6%9D%90\n        '
        data = optionaldict(template_id=template_id)
        return self._post('externalcontact/group_welcome_template/get', data=data)

    def del_group_welcome_template(self, template_id: str, agentid: Optional[int]=None) -> dict:
        if False:
            return 10
        '\n        删除入群欢迎语素材\n\n        企业可通过此API删除入群欢迎语素材，且仅能删除调用方自己创建的入群欢迎语素材。\n\n        详细请查阅企业微信官方文档 `删除入群欢迎语素材`_ 章节。\n\n        :param template_id: 群欢迎语的素材id\n        :param agentid: 授权方安装的应用agentid。仅旧的第三方多应用套件需要填此参数\n        :return: 响应数据\n\n        .. note::\n            **权限说明：**\n\n            - 需要使用 `客户联系secret`_ 或配置到 `可调用应用`_ 列表中的自建应用secret\n              来初始化 :py:class:`wechatpy.work.client.WeChatClient` 类。\n            - 第三方应用需具有“企业客户权限->客户联系->配置入群欢迎语素材”权限\n            - 仅可删除本应用创建的入群欢迎语素材\n\n        .. _删除入群欢迎语素材: https://work.weixin.qq.com/api/doc/90000/90135/\n            92366#%E5%88%A0%E9%99%A4%E5%85%A5%E7%BE%A4%E6%AC%A2%E8%BF%8E%E8\n            %AF%AD%E7%B4%A0%E6%9D%90\n        '
        data = optionaldict(template_id=template_id, agentid=agentid)
        return self._post('externalcontact/group_welcome_template/del', data=data)