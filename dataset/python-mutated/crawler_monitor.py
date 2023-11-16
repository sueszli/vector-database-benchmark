import datetime
import sys
sys.path.append('..')
from configure.settings import DBSelector
from configure.util import send_message_via_wechat

class Monitor:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def jsl_data_monitor(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        集思录数据监控\n        '
        db = DBSelector().get_mysql_conn('db_jisilu', type_='tencent-1c')
        cursor = db.cursor()
        date = datetime.datetime.now().strftime('%Y-%m-%d')
        sql = 'select `更新日期` from `tb_jsl_{}`'.format(date)
        count = 'select count(*) from `tb_jsl_{}`'.format(date)
        try:
            cursor.execute(sql)
            ret = cursor.fetchone()
            cursor.execute(count)
            count_ret = cursor.fetchone()
        except Exception as e:
            msg = '当天爬取集思录数据出错'
            send_message_via_wechat(msg)
        try:
            date_ = ret[0].split(' ')[0]
        except Exception as e:
            msg = '当天爬取集思录数据日期解析出错'
            send_message_via_wechat(msg)
            return
        if date_ != date:
            msg = '当天爬取集思录数据日期解析出错'
            send_message_via_wechat(msg)
            return
        if count_ret[0] < 200:
            msg = '当天爬取集思录数据条数出错'
            send_message_via_wechat(msg)
            return

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self.jsl_data_monitor()

def main():
    if False:
        for i in range(10):
            print('nop')
    app = Monitor()
    app.run()
if __name__ == '__main__':
    main()