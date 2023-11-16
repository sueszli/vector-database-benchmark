__author__ = 'Rocky'
'\nhttp://30daydo.com\nContact: weigesysu@qq.com\n'
import tushare as ts
import os
import matplotlib.pyplot as plt

class NewStockBreak:

    def __init__(self, start_date=20170101, end_date=20170401):
        if False:
            for i in range(10):
                print('nop')
        current = os.getcwd()
        folder = os.path.join(current, 'new_stock')
        if os.path.exists(folder) == False:
            os.mkdir(folder)
        os.chdir(folder)
        df0 = ts.get_stock_basics()
        self.bases = df0.sort_values('timeToMarket', ascending=False)
        self.cxg = self.bases[(self.bases['timeToMarket'] > start_date) & (self.bases['timeToMarket'] < end_date)]
        self.codes = self.cxg.index.values

    def calc_open_by_percent(self, code):
        if False:
            while True:
                i = 10
        cont = 100000000
        acutal_vol = self.bases.loc[code]['outstanding']
        all_vol = acutal_vol * cont
        df_k_data = ts.get_k_data(code)
        i = 1
        found = False
        df_k_data = df_k_data.sort_index(axis=0, ascending=True, by=['date'])
        while i < 365:
            try:
                s = df_k_data.iloc[i]
            except IndexError:
                print('single positional indexer is out-of-bounds')
                break
            except Exception as e:
                print(e)
                break
            else:
                if s['high'] != s['low']:
                    found = True
                    break
                i = i + 1
        if found:
            date_end = df_k_data.iloc[i]['date']
            date_start = df_k_data.iloc[0]['date']
            df3 = df_k_data[(df_k_data['date'] >= date_start) & (df_k_data['date'] <= date_end)]
            v_total_break = df3['volume'].sum()
            day = len(df3)
            rate = round(v_total_break * 100 * 100.0 / all_vol, 2)
        else:
            (rate, day) = (0, 0)
        return (rate, day)

    def calc_open_day(self, code):
        if False:
            for i in range(10):
                print('nop')
        cont = 100000000
        acutal_vol = self.bases[self.bases['code'] == code]['outstanding'].values[0]
        all_vol = acutal_vol * cont
        df1 = ts.get_k_data(code)
        if len(df1) < 3:
            return None
        start = df1['date'].values[0]
        print('Start day:', start)
        df2 = df1[(df1['close'] == df1['low']) & (df1['high'] == df1['low'])]
        print(self.bases[self.bases['code'] == code]['name'].values[0])
        end = df2['date'].values[-1]
        print('Break day', end)
        df3 = df1[(df1['date'] >= start) & (df1['date'] <= end)]
        v_total_break = df3['volume'].sum()
        l = len(df3)
        print(l)
        print(v_total_break)
        rate = v_total_break * 100 * 100.0 / all_vol
        print(round(rate, 6))
        return (rate, l)

    def testcase(self):
        if False:
            while True:
                i = 10
        result = []
        max_line = []
        k = []
        for i in self.codes:
            (t, l) = self.calc_open_day(i)
            if t is not None:
                result.append(t)
                max_line.append({i: l})
                k.append(l)
        x = range(len(result))
        plt.bar(x, result)
        plt.show()
        sum = 0
        for i in result:
            sum = sum + i
        avg = sum * 1.0 / len(result)
        print(avg)
        max_v = max(k)
        print(max_v)
        print(max_line)

    def getData(self, filename):
        if False:
            i = 10
            return i + 15
        result = []
        max_line = []
        k = []
        for i in self.codes:
            print(f'正处理{i}')
            name = self.bases.loc[i]['name']
            (rate, day) = self.calc_open_by_percent(i)
            if rate:
                result.append(rate)
                max_line.append([name, day, rate])
                k.append(day)
        with open(filename, 'w') as f:
            for x in max_line:
                f.write(x[0])
                f.write(';')
                f.write(str(x[1]))
                f.write(';')
                f.write(str(x[2]))
                f.write('\n')

def main():
    if False:
        for i in range(10):
            print('nop')
    obj = NewStockBreak(start_date=20200101, end_date=20200701)
    obj.getData('cxg.txt')
if __name__ == '__main__':
    main()