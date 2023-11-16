"""
@author:xda
@file:fund_share_monitor.py
@time:2021/01/27
"""
import sys
sys.path.append('..')
from configure.settings import DBSelector
from common.BaseService import BaseService
from fund.fund_share_crawl import ShareModel, FundBaseInfoModel, Fund
from sqlalchemy import and_

class ShareMonitor(Fund):

    def __init__(self):
        if False:
            return 10
        super(ShareMonitor, self).__init__()
        self.sess = self.get_session()()

    def query(self, code, date):
        if False:
            print('Hello World!')
        obj = self.sess.query(ShareModel).filter(and_(ShareModel.date <= date, ShareModel.code == code)).all()
        if obj:
            for i in obj:
                print(i.code)
                print(i.share)
                print(i.date)
                print('')
if __name__ == '__main__':
    app = ShareMonitor()
    code = '167302'
    date = '2021-01-26'
    app.query(code=code, date=date)