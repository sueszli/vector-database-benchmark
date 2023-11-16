import random
from datetime import datetime
from wechatpy.pay.api.base import BaseWeChatPayAPI

class WeChatCoupon(BaseWeChatPayAPI):

    def send(self, user_id, stock_id, op_user_id=None, device_info=None, out_trade_no=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        发放代金券\n\n        :param user_id: 用户在公众号下的 openid\n        :param stock_id: 代金券批次 ID\n        :param op_user_id: 可选，操作员账号，默认为商户号\n        :param device_info: 可选，微信支付分配的终端设备号\n        :param out_trade_no: 可选，商户订单号，需保持唯一性，默认自动生成\n        :return: 返回的结果信息\n        '
        if not out_trade_no:
            now = datetime.now()
            out_trade_no = f"{self.mch_id}{now.strftime('%Y%m%d%H%M%S')}{random.randint(1000, 10000)}"
        data = {'appid': self.appid, 'coupon_stock_id': stock_id, 'openid': user_id, 'openid_count': 1, 'partner_trade_no': out_trade_no, 'op_user_id': op_user_id, 'device_info': device_info, 'version': '1.0', 'type': 'XML'}
        return self._post('mmpaymkttransfers/send_coupon', data=data)

    def query_stock(self, stock_id, op_user_id=None, device_info=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        查询代金券批次\n\n        :param stock_id: 代金券批次 ID\n        :param op_user_id: 可选，操作员账号，默认为商户号\n        :param device_info: 可选，微信支付分配的终端设备号\n        :return: 返回的结果信息\n        '
        data = {'appid': self.appid, 'coupon_stock_id': stock_id, 'op_user_id': op_user_id, 'device_info': device_info, 'version': '1.0', 'type': 'XML'}
        return self._post('mmpaymkttransfers/query_coupon_stock', data=data)

    def query_coupon(self, coupon_id, user_id, op_user_id=None, device_info=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        查询代金券信息\n\n        :param coupon_id: 代金券 ID\n        :param user_id: 用户在公众号下的 openid\n        :param op_user_id: 可选，操作员账号，默认为商户号\n        :param device_info: 可选，微信支付分配的终端设备号\n        :return: 返回的结果信息\n        '
        data = {'coupon_id': coupon_id, 'openid': user_id, 'appid': self.appid, 'op_user_id': op_user_id, 'device_info': device_info, 'version': '1.0', 'type': 'XML'}
        return self._post('promotion/query_coupon', data=data)