__author__ = 'Rocky'
'\nhttp://30daydo.com\nEmail: weigesysu@qq.com\n'
import sys
sys.path.append('..')
from configure.settings import DBSelector
from common.BaseService import BaseService
from common.SecurityBase import StockBase

class StockDoctor(BaseService, StockBase):

    def __init__(self):
        if False:
            print('Hello World!')
        BaseService.__init__(self, f'log/{self.__class__.__name__}.log')
        StockBase.__init__(self)
        self.logger.info('start')
        self.DB = DBSelector()
        self.conn = self.DB.get_mysql_conn('db_stock', 'qq')
        self.cursor = self.conn.cursor()

    def check_blacklist(self, code):
        if False:
            for i in range(10):
                print('nop')
        cmd = 'select * from tb_blacklist where code=%s'
        self.cursor.execute(cmd, args=(code,))
        ret = self.cursor.fetchone()
        if ret:
            return True
        else:
            return False

    def north_east(self, code):
        if False:
            while True:
                i = 10
        north_east_area = ['黑龙江', '吉林', '辽宁']
        cmd = 'select area from tb_basic_info where code=%s'
        self.cursor.execute(cmd, args=(code,))
        ret = self.cursor.fetchone()
        if ret and ret in north_east_area:
            return True
        else:
            return False

    def get_code(self, name):
        if False:
            print('Hello World!')
        cmd = 'select code from tb_basic_info where name=%s'
        self.cursor.execute(cmd, args=name)
        ret = self.cursor.fetchone()
        return ret

    def diagnose(self, code):
        if False:
            i = 10
            return i + 15
        if not self.valid_code(code):
            raise ValueError('输入有误')
        issue = False
        if self.check_blacklist(code):
            self.logger.info('存在黑名单')
            issue = True
        if self.north_east(code):
            self.logger.info('是东北股')
            issue = True
        if issue:
            self.logger.info(f'{code} 问题股')

def main():
    if False:
        print('Hello World!')
    code = input('输入诊断个股的代码： ')
    doctor = StockDoctor()
    doctor.diagnose(code)
if __name__ == '__main__':
    main()