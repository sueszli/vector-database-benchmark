__author__ = 'Rocky'
'\nhttp://30daydo.com\nContact: weigesysu@qq.com\n'
import tushare as ts
import pandas as pd
import os, datetime, time, Queue
from toolkit import Toolkit
from threading import Thread
q = Queue.Queue()
pd.set_option('max_rows', None)
from configure.settings import get_engine
engine = get_engine('db_stock')

class filter_stock:

    def __init__(self, retry=5, local=False):
        if False:
            i = 10
            return i + 15
        if local:
            for i in range(retry):
                try:
                    self.bases_save = ts.get_stock_basics()
                    self.bases_save = self.bases_save.reset_index()
                    self.bases_save.to_csv('bases.csv')
                    self.bases_save.to_sql('bases', engine, if_exists='replace')
                    if self.bases_save:
                        break
                except Exception as e:
                    if i >= 4:
                        self.bases_save = pd.DataFrame()
                        exit()
                    continue
        else:
            self.bases_save = pd.read_sql('bases', engine, index_col='index')
            self.base = self.bases_save
        self.today = time.strftime('%Y-%m-%d', time.localtime())
        self.all_code = self.base['code'].values
        self.working_count = 0
        self.mystocklist = Toolkit.read_stock('mystock.csv')

    def save_data_excel(self):
        if False:
            return 10
        df = ts.get_stock_basics()
        df.to_csv(self.today + '.csv', encoding='gbk')
        df_x = pd.read_csv(self.today + '.csv', encoding='gbk')
        df_x.to_excel(self.today + '.xls', encoding='gbk')
        os.remove(self.today + '.csv')

    def insert_garbe(self):
        if False:
            i = 10
            return i + 15
        print('*' * 30)
        print('\n')

    def showInfo(self, df):
        if False:
            for i in range(10):
                print('nop')
        print('*' * 30)
        print('\n')
        print(df.info())
        print('*' * 30)
        print('\n')
        print(df.dtypes)
        self.insert_garbe()
        print(df.describe())

    def count_area(self, writeable=False):
        if False:
            for i in range(10):
                print('nop')
        count = self.base['area'].value_counts()
        print(count)
        print(type(count))
        if writeable:
            count.to_csv('各省的上市公司数目.csv')
        return count

    def get_area(self, area, writeable=False):
        if False:
            print('Hello World!')
        user_area = self.base[self.base['area'] == area]
        user_area.sort_values('timeToMarket', inplace=True, ascending=False)
        if writeable:
            filename = area + '.csv'
            user_area.to_csv(filename)
        return user_area

    def get_all_location(self):
        if False:
            i = 10
            return i + 15
        series = self.count_area()
        index = series.index
        for i in index:
            name = unicode(i)
            self.get_area(name, writeable=True)

    def fetch_new_ipo(self, start_time, writeable=False):
        if False:
            i = 10
            return i + 15
        df = self.base.loc[self.base['timeToMarket'] > start_time]
        df.sort_values('timeToMarket', inplace=True, ascending=False)
        if writeable == True:
            df.to_csv('New_IPO.csv')
        pe_av = df[df['pe'] != 0]['pe'].mean()
        pe_all_av = self.base[self.base['pe'] != 0]['pe'].mean()
        print(u'平均市盈率为 ', pe_av)
        print('A股的平均市盈率为 ', pe_all_av)
        return df

    def get_chengfenggu(self, writeable=False):
        if False:
            i = 10
            return i + 15
        s50 = ts.get_sz50s()
        if writeable == True:
            s50.to_excel('sz50.xls')
        list_s50 = s50['code'].values.tolist()
        return list_s50

    def drop_down_from_high(self, start, code):
        if False:
            i = 10
            return i + 15
        end_day = datetime.date(datetime.date.today().year, datetime.date.today().month, datetime.date.today().day)
        end_day = end_day.strftime('%Y-%m-%d')
        total = ts.get_k_data(code=code, start=start, end=end_day)
        high = total['high'].max()
        high_day = total.loc[total['high'] == high]['date'].values[0]
        print(high)
        print(high_day)
        current = total['close'].values[-1]
        print(current)
        percent = round((current - high) / high * 100, 2)
        print(percent)
        return percent

    def loop_each_cixin(self):
        if False:
            print('Hello World!')
        df = self.fetch_new_ipo(20170101, writeable=False)
        all_code = df['code'].values
        print(all_code)
        percents = []
        for each in all_code:
            print(each)
            percent = self.drop_down_from_high('2017-01-01', each)
            percents.append(percent)
        df['Drop_Down'] = percents
        df.sort_values('Drop_Down', ascending=True, inplace=True)
        df.to_csv(self.today + '_drop_Down_cixin.csv')

    def macd(self):
        if False:
            for i in range(10):
                print('nop')
        result = []
        for each_code in self.all_code:
            print(each_code)
            try:
                df_x = ts.get_k_data(code=each_code, start='2017-03-01')
            except:
                print("Can't get k_data")
                continue
            if len(df_x) < 11:
                print('no item')
                continue
            ma5 = df_x['close'][-5:].mean()
            ma10 = df_x['close'][-10:].mean()
            if ma5 > ma10:
                temp = [each_code, self.base[self.base['code'] == each_code]['name'].values[0]]
                print(temp)
                result.append(temp)
        print(result)
        print('Done')
        return result

    def get_all_code(self):
        if False:
            i = 10
            return i + 15
        return self.all_code

    def volume_calculate(self, codes):
        if False:
            return 10
        delta_day = 180 * 7 / 5
        end_day = datetime.date(datetime.date.today().year, datetime.date.today().month, datetime.date.today().day)
        start_day = end_day - datetime.timedelta(delta_day)
        start_day = start_day.strftime('%Y-%m-%d')
        end_day = end_day.strftime('%Y-%m-%d')
        print(start_day)
        print(end_day)
        result_m5_large = []
        result_m5_small = []
        for each_code in codes:
            try:
                df = ts.get_k_data(each_code, start=start_day, end=end_day)
                print(df)
            except Exception as e:
                print('Failed to get')
                print(e)
                continue
            if len(df) < 20:
                continue
            print(each_code)
            all_mean = df['volume'].mean()
            m5_volume_m = df['volume'][-5:].mean()
            m10_volume_m = df['volume'][-10:].mean()
            last_vol = df['volume'][-1]
            if m5_volume_m > 4.0 * all_mean:
                print('m5 > m_all_avg ')
                print(each_code)
                temp = self.base[self.base['code'] == each_code]['name'].values[0]
                print(temp)
                result_m5_large.append(each_code)
            if last_vol < m5_volume_m / 3.0:
                result_m5_small.append(each_code)
        return (result_m5_large, result_m5_large)

    def turnover_check(self):
        if False:
            i = 10
            return i + 15
        delta_day = 60 * 7 / 5
        end_day = datetime.date(datetime.date.today().year, datetime.date.today().month, datetime.date.today().day)
        start_day = end_day - datetime.timedelta(delta_day)
        start_day = start_day.strftime('%Y-%m-%d')
        end_day = end_day.strftime('%Y-%m-%d')
        print(start_day)
        print(end_day)
        for each_code in self.all_code:
            try:
                df = ts.get_hist_data(code=each_code, start=start_day, end=end_day)
            except:
                print('Failed to get data')
                continue
            mv5 = df['v_ma5'][-1]
            mv20 = df['v_ma20'][-1]
            mv_all = df['volume'].mean()

    def write_to_text(self):
        if False:
            print('Hello World!')
        print('On write')
        r = self.macd()
        filename = self.today + '-macd.csv'
        f = open(filename, 'w')
        for i in r:
            f.write(i[0])
            f.write(',')
            f.write(i[1])
            f.write('\n')
        f.close()

    def saveList(self, l, name):
        if False:
            i = 10
            return i + 15
        f = open(self.today + name + '.csv', 'w')
        if len(l) == 0:
            return False
        for i in l:
            f.write(i)
            f.write(',')
            name = self.base[self.base['code'] == i]['name'].values[0]
            f.write(name)
            f.write('\n')
        f.close()
        return True

    def read_csv(self):
        if False:
            while True:
                i = 10
        filename = self.today + '-macd.csv'
        df = pd.read_csv(filename)
        print(df)

    def own_drop_down(self):
        if False:
            print('Hello World!')
        for i in self.mystocklist:
            print(i)
            self.drop_down_from_high(code=i, start='2017-01-01')
            print('\n')

    def _break_line(self, codes, k_type):
        if False:
            while True:
                i = 10
        delta_day = 60 * 7 / 5
        end_day = datetime.date(datetime.date.today().year, datetime.date.today().month, datetime.date.today().day)
        start_day = end_day - datetime.timedelta(delta_day)
        start_day = start_day.strftime('%Y-%m-%d')
        end_day = end_day.strftime('%Y-%m-%d')
        print(start_day)
        print(end_day)
        all_break = []
        for i in codes:
            try:
                df = ts.get_hist_data(code=i, start=start_day, end=end_day)
                if len(df) == 0:
                    continue
            except Exception as e:
                print(e)
                continue
            else:
                self.working_count = self.working_count + 1
                current = df['close'][0]
                ma5 = df['ma5'][0]
                ma10 = df['ma10'][0]
                ma20 = df['ma20'][0]
                ma_dict = {'5': ma5, '10': ma10, '20': ma20}
                ma_x = ma_dict[k_type]
                if current < ma_x:
                    print('破位')
                    print(i, ' current: ', current)
                    print(self.base[self.base['code'] == i]['name'].values[0], ' ')
                    print('holding place: ', ma_x)
                    print('Break MA', k_type, '\n')
                    all_break.append(i)
        return all_break

    def break_line(self, code, k_type='20', writeable=False, mystock=False):
        if False:
            for i in range(10):
                print('nop')
        all_break = self._break_line(code, k_type)
        l = len(all_break)
        beaking_rate = l * 1.0 / self.working_count * 100
        print('how many break: ', l)
        print('break Line rate ', beaking_rate)
        if mystock == False:
            name = '_all_'
        else:
            name = '_my__'
        if writeable:
            f = open(self.today + name + 'break_line_' + k_type + '.csv', 'w')
            f.write('Breaking rate: %f\n\n' % beaking_rate)
            f.write('\n'.join(all_break))
            f.close()

    def _break_line_thread(self, codes, k_type='5'):
        if False:
            return 10
        delta_day = 60 * 7 / 5
        end_day = datetime.date(datetime.date.today().year, datetime.date.today().month, datetime.date.today().day)
        start_day = end_day - datetime.timedelta(delta_day)
        start_day = start_day.strftime('%Y-%m-%d')
        end_day = end_day.strftime('%Y-%m-%d')
        print(start_day)
        print(end_day)
        all_break = []
        for i in codes:
            try:
                df = ts.get_hist_data(code=i, start=start_day, end=end_day)
                if len(df) == 0:
                    continue
            except Exception as e:
                print(e)
                continue
            else:
                self.working_count = self.working_count + 1
                current = df['close'][0]
                ma5 = df['ma5'][0]
                ma10 = df['ma10'][0]
                ma20 = df['ma20'][0]
                ma_dict = {'5': ma5, '10': ma10, '20': ma20}
                ma_x = ma_dict[k_type]
                if current > ma_x:
                    print(i, ' current: ', current)
                    print(self.base[self.base['code'] == i]['name'].values[0], ' ')
                    print('Break MA', k_type, '\n')
                    all_break.append(i)
        q.put(all_break)

    def multi_thread_break_line(self, ktype='20'):
        if False:
            for i in range(10):
                print('nop')
        total = len(self.all_code)
        thread_num = 10
        delta = total / thread_num
        delta_left = total % thread_num
        t = []
        i = 0
        for i in range(thread_num):
            sub_code = self.all_code[i * delta:(i + 1) * delta]
            t_temp = Thread(target=self._break_line_thread, args=(sub_code, ktype))
            t.append(t_temp)
        if delta_left != 0:
            sub_code = self.all_code[i * delta:i * delta + delta_left]
            t_temp = Thread(target=self._break_line_thread, args=(sub_code, ktype))
            t.append(t_temp)
        for i in range(len(t)):
            t[i].start()
        for j in range(len(t)):
            t[j].join()
        result = []
        print('working done')
        while not q.empty():
            result.append(q.get())
        ff = open(self.today + '_high_m%s.csv' % ktype, 'w')
        for kk in result:
            print(kk)
            for k in kk:
                ff.write(k)
                ff.write(',')
                ff.write(self.base[self.base['code'] == k]['name'].values[0])
                ff.write('\n')
        ff.close()

    def relation(self):
        if False:
            return 10
        sh_index = ts.get_k_data('000001', index=True, start='2012-01-01')
        sh = sh_index['close'].values
        print(sh)
        vol_close = sh_index.corr()
        print(vol_close)
        "\n        sz_index=ts.get_k_data('399001',index=True)\n        sz=sz_index['close'].values\n        print(sz)\n\n        cy_index=ts.get_k_data('399006',index=True)\n        s1=Series(sh)\n        s2=Series(sz)\n        print(s1.corr(s2))\n        "

    def profit(self):
        if False:
            while True:
                i = 10
        df_2016 = ts.get_report_data(2016, 4)
        df_2015 = ts.get_report_data(2015, 4)
        df_2016.to_excel('2016_report.xls')
        df_2015.to_excel('2015_report.xls')
        code_2015_lost = df_2015[df_2015['net_profits'] < 0]['code'].values
        code_2016_lost = df_2016[df_2016['net_profits'] < 0]['code'].values
        print(code_2015_lost)
        print(code_2016_lost)
        two_year_lost = []
        for i in code_2015_lost:
            if i in code_2016_lost:
                print(i)
                two_year_lost.append(i)
        self.saveList(two_year_lost, 'st_dangours.csv')

    def mydaily_check(self):
        if False:
            print('Hello World!')
        self.break_line(self.mystocklist, k_type='5', writeable=True, mystock=True)

    def all_stock(self):
        if False:
            i = 10
            return i + 15
        self.multi_thread_break_line('20')

def get_break_bvps():
    if False:
        while True:
            i = 10
    base_info = ts.get_stock_basics()
    current_prices = ts.get_today_all()
    current_prices[current_prices['code'] == '000625']['trade'].values[0]
    base_info.loc['000625']['bvps']

def main():
    if False:
        while True:
            i = 10
    folder = os.path.join(os.path.dirname(__file__), 'data')
    if os.path.exists(folder) == False:
        os.mkdir(folder)
    os.chdir(folder)
    obj = filter_stock(local=True)
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    print(start_time)
    main()
    end_time = datetime.datetime.now()
    print(end_time)
    print('time use : ', (end_time - start_time).seconds)