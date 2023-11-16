"""
Created on 2016年9月25日
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""
import six
import pandas as pd
import requests
import time
from threading import Thread
from tushare.trader import vars as vs
from tushare.trader import utils
from tushare.util import upass as up
from tushare.util.upass import set_broker

class TraderAPI(object):
    """
    股票实盘交易接口
    提醒：本文涉及的思路和内容仅限于量化投资及程序化交易的研究与尝试，不作为个人或机构常规程序化交易的依据，
    不对实盘的交易风险和政策风险产生的影响负责，如有问题请与我联系。
    投资有风险，下单须谨慎。
    """

    def __init__(self, broker=''):
        if False:
            return 10
        if broker == '':
            return None
        self.broker = broker
        self.trade_prefix = vs.CSC_PREFIX % (vs.P_TYPE['https'], vs.DOMAINS['csc'], vs.PAGES['csclogin'])
        self.heart_active = True
        self.s = requests.session()
        if six.PY2:
            self.heart_thread = Thread(target=self.send_heartbeat)
            self.heart_thread.setDaemon(True)
        else:
            self.heart_thread = Thread(target=self.send_heartbeat, daemon=True)

    def login(self):
        if False:
            i = 10
            return i + 15
        self.s.headers.update(vs.AGENT)
        self.s.get(vs.CSC_PREFIX % (vs.P_TYPE['https'], vs.DOMAINS['csc'], vs.PAGES['csclogin']))
        res = self.s.get(vs.V_CODE_URL % (vs.P_TYPE['https'], vs.DOMAINS['cscsh'], vs.PAGES['vimg']))
        if self._login(utils.get_vcode('csc', res)) is False:
            print('请确认账号或密码是否正确 ，或券商服务器是否处于维护中。 ')
        self.keepalive()

    def _login(self, v_code):
        if False:
            print('Hello World!')
        brokerinfo = up.get_broker(self.broker)
        user = brokerinfo['user'][0]
        login_params = dict(inputid=user, j_username=user, j_inputid=user, AppendCode=v_code, isCheckAppendCode='false', logined='false', f_tdx='', j_cpu='', j_password=brokerinfo['passwd'][0])
        logined = self.s.post(vs.CSC_LOGIN_ACTION % (vs.P_TYPE['https'], vs.DOMAINS['csc']), params=login_params)
        if logined.text.find(u'消息中心') != -1:
            return True
        return False

    def keepalive(self):
        if False:
            print('Hello World!')
        if self.heart_thread.is_alive():
            self.heart_active = True
        else:
            self.heart_thread.start()

    def send_heartbeat(self):
        if False:
            while True:
                i = 10
        while True:
            if self.heart_active:
                try:
                    response = self.heartbeat()
                    self.check_account_live(response)
                except:
                    self.login()
                time.sleep(100)
            else:
                time.sleep(10)

    def heartbeat(self):
        if False:
            return 10
        return self.baseinfo

    def exit(self):
        if False:
            i = 10
            return i + 15
        self.heart_active = False

    def buy(self, stkcode, price=0, count=0, amount=0):
        if False:
            print('Hello World!')
        '\n    买入证券\n        params\n        ---------\n        stkcode:股票代码，string\n        pricce:委托价格，int\n        count:买入数量\n        amount:买入金额\n        '
        jsonobj = utils.get_jdata(self._trading(stkcode, price, count, amount, 'B', 'buy'))
        res = True if jsonobj['result'] == 'true' else False
        return res

    def sell(self, stkcode, price=0, count=0, amount=0):
        if False:
            print('Hello World!')
        '\n    卖出证券\n        params\n        ---------\n        stkcode:股票代码，string\n        pricce:委托价格，int\n        count:卖出数量\n        amount:卖出金额\n        '
        jsonobj = utils.get_jdata(self._trading(stkcode, price, count, amount, 'S', 'sell'))
        res = True if jsonobj['result'] == 'true' else False
        return res

    def _trading(self, stkcode, price, count, amount, tradeflag, tradetype):
        if False:
            for i in range(10):
                print('nop')
        txtdata = self.s.get(vs.TRADE_CHECK_URL % (vs.P_TYPE['https'], vs.DOMAINS['csc'], vs.PAGES['tradecheck'], tradeflag, stkcode, tradetype, utils.nowtime_str()))
        jsonobj = utils.get_jdata(txtdata)
        list = jsonobj['returnList'][0]
        secuid = list['buysSecuid']
        fundavl = list['fundavl']
        stkname = list['stkname']
        if secuid is not None:
            if tradeflag == 'B':
                buytype = vs.BUY
                count = count if count else amount // price // 100 * 100
            else:
                buytype = vs.SELL
                count = count if count else amount // price
            tradeparams = dict(stkname=stkname, stkcode=stkcode, secuid=secuid, buytype=buytype, bsflag=tradeflag, maxstkqty='', buycount=count, buyprice=price, _=utils.nowtime_str())
            tradeResult = self.s.post(vs.TRADE_URL % (vs.P_TYPE['https'], vs.DOMAINS['csc'], vs.PAGES['trade']), params=tradeparams)
            return tradeResult
        return None

    def position(self):
        if False:
            print('Hello World!')
        '\n    获取持仓列表\n        return:DataFrame\n        ----------------------\n        stkcode:证券代码\n        stkname:证券名称\n        stkqty :证券数量\n        stkavl :可用数量\n        lastprice:最新价格\n        costprice:成本价\n        income :参考盈亏（元）\n        '
        return self._get_position()

    def _get_position(self):
        if False:
            i = 10
            return i + 15
        self.s.headers.update(vs.AGENT)
        txtdata = self.s.get(vs.BASE_URL % (vs.P_TYPE['https'], vs.DOMAINS['csc'], vs.PAGES['position']))
        jsonobj = utils.get_jdata(txtdata)
        df = pd.DataFrame(jsonobj['data'], columns=vs.POSITION_COLS)
        return df

    def entrust_list(self):
        if False:
            for i in range(10):
                print('nop')
        '\n       获取委托单列表\n       return:DataFrame\n       ----------\n       ordersno:委托单号\n       stkcode:证券代码\n       stkname:证券名称\n       bsflagState:买卖标志\n       orderqty:委托数量\n       matchqty:成交数量\n       orderprice:委托价格\n       operdate:交易日期\n       opertime:交易时间\n       orderdate:下单日期\n       state:状态\n        '
        txtdata = self.s.get(vs.ENTRUST_LIST_URL % (vs.P_TYPE['https'], vs.DOMAINS['csc'], vs.PAGES['entrustlist'], utils.nowtime_str()))
        jsonobj = utils.get_jdata(txtdata)
        df = pd.DataFrame(jsonobj['data'], columns=vs.ENTRUST_LIST_COLS)
        return df

    def deal_list(self, begin=None, end=None):
        if False:
            i = 10
            return i + 15
        '\n    获取成交列表\n        params\n        -----------\n        begin:开始日期  YYYYMMDD\n        end:结束日期  YYYYMMDD\n        \n        return: DataFrame\n        -----------\n        ordersno:委托单号\n        matchcode:成交编号\n        trddate:交易日期\n        matchtime:交易时间\n        stkcode:证券代码\n        stkname:证券名称\n        bsflagState:买卖标志\n        orderprice:委托价格\n        matchprice:成交价格\n        orderqty:委托数量\n        matchqty:成交数量\n        matchamt:成交金额\n        '
        daterange = ''
        if (begin is None) & (end is None):
            selecttype = 'intraDay'
        else:
            daterange = vs.DEAL_DATE_RANGE % (begin, end)
            selecttype = 'all'
        txtdata = self.s.get(vs.DEAL_LIST_URL % (vs.P_TYPE['https'], vs.DOMAINS['csc'], vs.PAGES['deallist'], selecttype, daterange, utils.nowtime_str()))
        jsonobj = utils.get_jdata(txtdata)
        df = pd.DataFrame(jsonobj['data'], columns=vs.DEAL_LIST_COLS)
        return df

    def cancel(self, ordersno='', orderdate=''):
        if False:
            i = 10
            return i + 15
        '\n                 撤单\n        params\n        -----------\n        ordersno:委托单号，多个以逗号分隔，e.g. 1866,1867\n        orderdata:委托日期 YYYYMMDD，多个以逗号分隔，对应委托单好\n        return\n        ------------\n        string\n        '
        if (ordersno != '') & (orderdate != ''):
            params = dict(ordersno=ordersno, orderdate=orderdate, _=utils.nowtime_str())
            result = self.s.post(vs.CANCEL_URL % (vs.P_TYPE['https'], vs.DOMAINS['csc'], vs.PAGES['cancel']), params=params)
            jsonobj = utils.get_jdata(result.text)
            return jsonobj['msgMap']['ResultSucess']
        return None

    def baseinfo(self):
        if False:
            while True:
                i = 10
        '\n    获取帐户基本信息\n        return: Series\n        -------------\n        fundid:帐户ID\n        gpsz: 股票市值\n        fundvalue:基金市值\n        jihelicai:集合理财\n        fundbal:帐户余额\n        marketvalue:总资产\n        fundavl:可用余额\n        daixiao:代销份额\n        otc:OTC份额\n        '
        return self._get_baseinfo()

    def _get_baseinfo(self):
        if False:
            i = 10
            return i + 15
        self.s.headers.update(vs.AGENT)
        txtdata = self.s.get(vs.BASE_URL % (vs.P_TYPE['https'], vs.DOMAINS['csc'], vs.PAGES['baseInfo']))
        jsonobj = utils.get_jdata(txtdata)
        stkdata = jsonobj['data']['moneytype0']
        stkdata['fundid'] = jsonobj['fundid']
        return pd.Series(stkdata)

    def check_login_status(self, return_data):
        if False:
            return 10
        if hasattr(return_data, 'get') and return_data.get('error_no') == '-1':
            raise NotLoginError

class NotLoginError(Exception):

    def __init__(self, result=None):
        if False:
            while True:
                i = 10
        super(NotLoginError, self).__init__()
        self.result = result

    def heartbeat(self):
        if False:
            for i in range(10):
                print('nop')
        return self.baseinfo