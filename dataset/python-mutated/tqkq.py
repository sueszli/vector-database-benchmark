__author__ = 'yanqiong'
from typing import Optional
from tqsdk.tradeable.otg.base_otg import BaseOtg
from tqsdk.tradeable.mixin import FutureMixin, StockMixin

class TqKq(BaseOtg, FutureMixin):
    """天勤快期模拟账户类"""

    def __init__(self, td_url: Optional[str]=None, number: Optional[int]=None):
        if False:
            return 10
        '\n        创建快期模拟账户实例\n\n        快期模拟的账户和交易信息可以在快期专业版查看，可以点击 `快期专业版 <https://www.shinnytech.com/qpro/>`_ 进行下载\n\n        Args:\n            td_url (str): [可选]指定交易服务器的地址, 默认使用快期账户对应的交易服务地址\n\n            number (int): [可选]模拟交易账号编号, 默认为主模拟账号, 可以通过指定 1~99 的数字来使用辅模拟帐号, 各个帐号的数据完全独立, 使用该功能需要购买专业版的权限, 且对应的辅账户可以在快期专业版上登录并进行观察\n\n        Example1::\n\n            from tqsdk import TqApi, TqAuth, TqKq\n\n            tq_kq = TqKq()\n            api = TqApi(account=tq_kq, auth=TqAuth("快期账户", "账户密码"))\n            quote = api.get_quote("SHFE.cu2206")\n            print(quote)\n            # 下单限价单\n            order = api.insert_order(symbol="SHFE.cu2206", direction=\'BUY\', offset=\'OPEN\', limit_price=quote.last_price, volume=1)\n            while order.status == \'ALIVE\':\n                api.wait_update()\n                print(order)  # 打印委托单信息\n\n            print(tq_kq.get_account())  # 打印快期模拟账户信息\n\n            print(tq_kq.get_position("SHFE.cu2206"))  # 打印持仓信息\n\n            for trade in order.trade_records.values():\n                print(trade)  # 打印委托单对应的成交信息\n            api.close()\n\n        Example2::\n\n            from tqsdk import TqApi, TqAuth, TqKq, TqMultiAccount\n\n            # 创建快期模拟账户和辅模拟账户001\n            tq_kq = TqKq()\n            tq_kq001= TqKq(number=1)\n\n            # 使用多账户模块同时登录这两个模拟账户\n            api = TqApi(account=TqMultiAccount([tq_kq,tq_kq001]), auth=TqAuth("快期账户", "账户密码"))\n\n            print(tq_kq.get_account())  # 打印快期模拟账户信息\n\n            print(tq_kq001.get_account())  # 打印快期模拟001账户信息\n\n            api.close()\n\n\n        '
        super().__init__('快期模拟', str(number) if number else '', '', td_url=td_url)
        self._account_no = number

    @property
    def _account_name(self):
        if False:
            return 10
        if self._account_no:
            return f'{self._api._auth._user_name}:{self._account_no:03d}'
        else:
            return self._api._auth._user_name

    @property
    def _account_info(self):
        if False:
            return 10
        info = super(TqKq, self)._account_info
        info.update({'account_type': self._account_type})
        return info

    @property
    def _account_auth(self):
        if False:
            i = 10
            return i + 15
        return {'feature': 'tq_ma' if self._account_no else None, 'account_id': None, 'auto_add': False}

    def _update_otg_info(self, api):
        if False:
            return 10
        self._account_id = f'{api._auth._auth_id}{self._account_no:03d}' if self._account_no else api._auth._auth_id
        self._password = f'shinnytech{self._account_no:03d}' if self._account_no else api._auth._auth_id
        super(TqKq, self)._update_otg_info(api)

class TqKqStock(BaseOtg, StockMixin):
    """天勤实盘类"""

    def __init__(self, td_url: Optional[str]=None, number: Optional[int]=None):
        if False:
            return 10
        '\n        创建快期股票模拟账户实例\n\n        快期股票模拟为专业版功能，可以点击 `天勤量化专业版 <https://www.shinnytech.com/tqsdk_professional/>`_ 申请试用或购买\n\n        Args:\n            td_url (str): [可选]指定交易服务器的地址, 默认使用快期账户对应的交易服务地址\n\n            number (int): [可选]模拟交易账号编号, 默认为主模拟账号, 可以通过指定 1~99 的数字来使用辅模拟帐号, 各个帐号的数据完全独立\n\n        Example::\n\n            from tqsdk import TqApi, TqAuth, TqKqStock, TqChan\n\n            tq_kq_stock = TqKqStock()\n            api = TqApi(account=tq_kq_stock, auth=TqAuth("快期账户", "账户密码"))\n            quote = api.get_quote("SSE.688529")\n            print(quote)\n            # 下单限价单\n            order = api.insert_order("SSE.688529", volume=200, direction="BUY", limit_price=quote.ask_price1)\n            while order.status == \'ALIVE\':\n                api.wait_update()\n                print(order)  # 打印委托单信息\n\n            print(tq_kq_stock.get_account())  # 打印快期股票模拟账户信息\n\n            print(tq_kq_stock.get_position("SSE.688529"))  # 打印持仓信息\n\n            for trade in order.trade_records.values():\n                print(trade)  # 打印委托单对应的成交信息\n            api.close()\n\n        '
        super().__init__('快期股票模拟', str(number) if number else '', '', td_url=td_url)
        self._account_no = number

    @property
    def _account_name(self):
        if False:
            while True:
                i = 10
        if self._account_no:
            return f'{self._api._auth._user_name}_stock:{self._account_no:03d}'
        else:
            return self._api._auth._user_name + '_stock'

    @property
    def _account_info(self):
        if False:
            while True:
                i = 10
        info = super(TqKqStock, self)._account_info
        info.update({'account_type': self._account_type})
        return info

    @property
    def _account_auth(self):
        if False:
            while True:
                i = 10
        return {'feature': 'tq_ma' if self._account_no else None, 'account_id': self._auth_account_id, 'auto_add': False}

    def _update_otg_info(self, api):
        if False:
            return 10
        self._auth_account_id = api._auth._auth_id + '-sim-securities'
        self._account_id = f'{api._auth._auth_id}{self._account_no:03d}-sim-securities' if self._account_no else self._auth_account_id
        self._password = f'shinnytech{self._account_no:03d}' if self._account_no else api._auth._auth_id
        super(TqKqStock, self)._update_otg_info(api)