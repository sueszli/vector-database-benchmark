__author__ = 'mayanqiong'
from typing import Optional
from abc import ABC, abstractmethod
from tqsdk.baseModule import TqModule

class Tradeable(ABC, TqModule):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._account_key = self._get_account_key()

    def _get_account_key(self):
        if False:
            for i in range(10):
                print('nop')
        return str(id(self))

    @property
    @abstractmethod
    def _account_name(self):
        if False:
            return 10
        raise NotImplementedError

    @property
    def _account_info(self):
        if False:
            print('Hello World!')
        return {'account_key': self._account_key, 'account_name': self._account_name}

    @property
    def _account_auth(self):
        if False:
            print('Hello World!')
        return {'feature': None, 'account_id': None, 'auto_add': False}

    def _is_self_trade_pack(self, pack):
        if False:
            while True:
                i = 10
        '是否是当前交易实例应该处理的交易包'
        if pack['aid'] in ['insert_order', 'cancel_order', 'set_risk_management_rule']:
            assert 'account_key' in pack, '发给交易请求的包必须包含 account_key'
            if pack['account_key'] != self._account_key:
                return False
            else:
                pack.pop('account_key', None)
                return True
        return False

    def _connect_td(self, api, index: int) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return None