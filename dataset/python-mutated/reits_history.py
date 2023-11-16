import datetime
import time
import sys
sys.path.append('..')
import requests

class ReitsHistoryData:

    def __init__(self):
        if False:
            return 10
        self.db = SQLCls()

    def crawl(self, code, ts):
        if False:
            while True:
                i = 10
        cookies = {'device_id': '30c150d8bba6b59a776c2e783ab3baf4', 's': 'by1hv4ciih', '__utmz': '1.1645204918.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none)', 'bid': 'a8ec0ec01035c8be5606c595aed718d4_kztd4jue', 'xq_is_login': '1', 'u': '1733473480', 'Hm_lvt_fe218c11eab60b6ab1b6f84fb38bcc4a': '1666192921,1667440399', '__utma': '1.1751771128.1645204918.1667539442.1667824918.36', 'xq_a_token': '2cb229cbb333f6f67f87f92d753ac51667d886ba', 'xqat': '2cb229cbb333f6f67f87f92d753ac51667d886ba', 'xq_id_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ1aWQiOjE3MzM0NzM0ODAsImlzcyI6InVjIiwiZXhwIjoxNjcwNDY2MjIwLCJjdG0iOjE2Njc4NzQyMjA2ODMsImNpZCI6ImQ5ZDBuNEFadXAifQ.IeeTTZxq_6DL314VZn74eUBpyOSJ_rwajC1kA52oouGd1-RyVqVd8SEtLlTWr99f87BA1MC5djZlMQ4lyZVYUf4Jj8P2lEZl4MuP9_rRkpQs47Z7_ey0RhAGtP8Frcv3SSjz11gl_nqSaVTarCuocFGXtaET7DLjlWZeeDvdfyXE0iUXkH28N5l5PBhVhEUZUFI6zQDSubgW252JERVkoJNa3tQaDEbPfRAvDIFjpGkM9kBAPZYVi7LSlAzTzGEuBUodiVmMmeiD9xv3VpjCpfpQv6AbK4NPe8HfwIMulA8y5M3hYdAILBqqCz6D1iensibnWXZo0xnobnpGAQzp8A', 'xq_r_token': 'a5d6827d5621ceaac46ad8a5334c2210ba66952f', 'Hm_lvt_1db88642e346389874251b5a1eded6e3': '1667437811,1667531174,1667791869,1667874222', 'Hm_lpvt_1db88642e346389874251b5a1eded6e3': '1667889958', 'acw_tc': '2760779716678949087306123e2da0f6e6088fa3c13cf0f5b08312a995a828'}
        headers = {'authority': 'stock.xueqiu.com', 'pragma': 'no-cache', 'cache-control': 'no-cache', 'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="98", "Google Chrome";v="98"', 'sec-ch-ua-mobile': '?0', 'sec-ch-ua-platform': '"Linux"', 'upgrade-insecure-requests': '1', 'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36', 'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9', 'sec-fetch-site': 'none', 'sec-fetch-mode': 'navigate', 'sec-fetch-user': '?1', 'sec-fetch-dest': 'document', 'accept-language': 'zh,en;q=0.9,en-US;q=0.8,zh-CN;q=0.7,zh-TW;q=0.6'}
        response = requests.get('https://stock.xueqiu.com/v5/stock/chart/kline.json?symbol={}&begin={}&period=day&type=before&count=-100&indicator=kline,pe,pb,ps,pcf,market_capital,agt,ggt,balance'.format(code, ts), cookies=cookies, headers=headers)
        return response.json()

    def gen_params(self):
        if False:
            for i in range(10):
                print('nop')
        fmt = '%Y-%m-%d %H:%M:%S'
        current = '2021-05-30 00:00:00'
        now = '2023-05-08 00:00:00'
        delta = 100
        result = []
        while current < now:
            current = datetime.datetime.strptime(current, fmt)
            current = (current + datetime.timedelta(days=delta)).strftime(fmt)
            b = time.strptime(current, fmt)
            arg = int(time.mktime(b)) * 1000
            print(arg)
            result.append(arg)
        return result

    def convert_int(self, t):
        if False:
            while True:
                i = 10
        try:
            t = int(t)
        except Exception as e:
            return None
        else:
            return t

    def convert_float(self, f):
        if False:
            print('Hello World!')
        try:
            f = float(f)
        except Exception as e:
            return None
        else:
            return f

    def parse(self, js_data, code):
        if False:
            print('Hello World!')
        items = js_data.get('data', {}).get('item', [])
        for item in items:
            date = time.strftime('%Y-%m-%d', time.localtime(int(item[0] / 1000)))
            print(date)
            volume = self.convert_int(item[1])
            open_p = self.convert_float(item[2])
            high = self.convert_float(item[3])
            low = self.convert_float(item[4])
            close = self.convert_float(item[5])
            chg = self.convert_float(item[6])
            percent = self.convert_float(item[7])
            turnoverrate = self.convert_float(item[8])
            amount = self.convert_float(item[9])
            data = (date, volume, open_p, high, low, close, chg, percent, turnoverrate, amount, code)
            self.db.insert_data(data)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        code_list = ['SZ180501', 'SH508018']
        for code in code_list:
            print('crawling code {}'.format(code))
            ts_list = self.gen_params()
            for ts in ts_list:
                js = self.crawl(code, ts)
                self.parse(js, code)

class SQLCls:

    def __init__(self):
        if False:
            while True:
                i = 10
        from configure.settings import DBSelector
        self.conn = DBSelector().get_mysql_conn('db_reits', 'tencent-1c')
        self.cursor = self.conn.cursor()

    def insert_data(self, item):
        if False:
            for i in range(10):
                print('nop')
        sql = 'insert into `reits_history` (date,volume,open,high,low,close,chg,percent,turnoverrate,amount,code) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
        try:
            self.cursor.execute(sql, item)
        except Exception as e:
            print(e)
            self.conn.rollback()
        else:
            self.conn.commit()

def main():
    if False:
        for i in range(10):
            print('nop')
    app = ReitsHistoryData()
    app.run()
if __name__ == '__main__':
    main()