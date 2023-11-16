import sys
import pymysql
import requests
import json
import re
from bs4 import BeautifulSoup
'\n类说明:获取财务数据\n\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nZhihu:\n\thttps://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-08-31\n'

class FinancialData:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.server = 'http://quotes.money.163.com/'
        self.cwnb = 'http://quotes.money.163.com/hkstock/cwsj_'
        self.cwzb_dict = {'EPS': '基本每股收益', 'EPS_DILUTED': '摊薄每股收益', 'GROSS_MARGIN': '毛利率', 'CAPITAL_ADEQUACY': '资本充足率', 'LOANS_DEPOSITS': '贷款回报率', 'ROTA': '总资产收益率', 'ROEQUITY': '净资产收益率', 'CURRENT_RATIO': '流动比率', 'QUICK_RATIO': '速动比率', 'ROLOANS': '存贷比', 'INVENTORY_TURNOVER': '存货周转率', 'GENERAL_ADMIN_RATIO': '管理费用比率', 'TOTAL_ASSET2TURNOVER': '资产周转率', 'FINCOSTS_GROSSPROFIT': '财务费用比率', 'TURNOVER_CASH': '销售现金比率', 'YEAREND_DATE': '报表日期'}
        self.lrb_dict = {'TURNOVER': '总营收', 'OPER_PROFIT': '经营利润', 'PBT': '除税前利润', 'NET_PROF': '净利润', 'EPS': '每股基本盈利', 'DPS': '每股派息', 'INCOME_INTEREST': '利息收益', 'INCOME_NETTRADING': '交易收益', 'INCOME_NETFEE': '费用收益', 'YEAREND_DATE': '报表日期'}
        self.fzb_dict = {'FIX_ASS': '固定资产', 'CURR_ASS': '流动资产', 'CURR_LIAB': '流动负债', 'INVENTORY': '存款', 'CASH': '现金及银行存结', 'OTHER_ASS': '其他资产', 'TOTAL_ASS': '总资产', 'TOTAL_LIAB': '总负债', 'EQUITY': '股东权益', 'CASH_SHORTTERMFUND': '库存现金及短期资金', 'DEPOSITS_FROM_CUSTOMER': '客户存款', 'FINANCIALASSET_SALE': '可供出售之证券', 'LOAN_TO_BANK': '银行同业存款及贷款', 'DERIVATIVES_LIABILITIES': '金融负债', 'DERIVATIVES_ASSET': '金融资产', 'YEAREND_DATE': '报表日期'}
        self.llb_dict = {'CF_NCF_OPERACT': '经营活动产生的现金流', 'CF_INT_REC': '已收利息', 'CF_INT_PAID': '已付利息', 'CF_INT_REC': '已收股息', 'CF_DIV_PAID': '已派股息', 'CF_INV': '投资活动产生现金流', 'CF_FIN_ACT': '融资活动产生现金流', 'CF_BEG': '期初现金及现金等价物', 'CF_CHANGE_CSH': '现金及现金等价物净增加额', 'CF_END': '期末现金及现金等价物', 'CF_EXCH': '汇率变动影响', 'YEAREND_DATE': '报表日期'}
        self.table_dict = {'cwzb': self.cwzb_dict, 'lrb': self.lrb_dict, 'fzb': self.fzb_dict, 'llb': self.llb_dict}
        self.headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8', 'Accept-Encoding': 'gzip, deflate', 'Accept-Language': 'zh-CN,zh;q=0.8', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.109 Safari/537.36'}
    '\n\t函数说明:获取股票页面信息\n\n\tAuthor:\n\t\tJack Cui\n\tParameters:\n\t    url - 股票财务数据界面地址\n\tReturns:\n\t    name - 股票名\n\t    table_name_list - 财务报表名称\n\t    table_date_list - 财务报表年限\n\t    url_list - 财务报表查询连接\n\tBlog:\n\t\thttp://blog.csdn.net/c406495762\n\tZhihu:\n\t\thttps://www.zhihu.com/people/Jack--Cui/\n\tModify:\n\t\t2017-08-31\n\t'

    def get_informations(self, url):
        if False:
            for i in range(10):
                print('nop')
        req = requests.get(url=url, headers=self.headers)
        req.encoding = 'utf-8'
        html = req.text
        page_bf = BeautifulSoup(html, 'lxml')
        name = page_bf.find_all('span', class_='name')[0].string
        table_name_list = []
        table_date_list = []
        each_date_list = []
        url_list = []
        table_name = page_bf.find_all('div', class_='titlebar3')
        for each_table_name in table_name:
            table_name_list.append(each_table_name.span.string)
            for each_table_date in each_table_name.div.find_all('select', id=re.compile('.+1$')):
                url_list.append(re.findall('(\\w+)1', each_table_date.get('id'))[0])
                for each_date in each_table_date.find_all('option'):
                    each_date_list.append(each_date.string)
                table_date_list.append(each_date_list)
                each_date_list = []
        return (name, table_name_list, table_date_list, url_list)
    '\n\t函数说明:财务报表入库\n\n\tAuthor:\n\t\tJack Cui\n\tParameters:\n\t    name - 股票名\n\t    table_name_list - 财务报表名称\n\t    table_date_list - 财务报表年限\n\t    url_list - 财务报表查询连接\n\tReturns:\n\t\t无\n\tBlog:\n\t\thttp://blog.csdn.net/c406495762\n\tZhihu:\n\t\thttps://www.zhihu.com/people/Jack--Cui/\n\tModify:\n\t\t2017-08-31\n\t'

    def insert_tables(self, name, table_name_list, table_date_list, url_list):
        if False:
            for i in range(10):
                print('nop')
        conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='yourpasswd', db='financialdata', charset='utf8')
        cursor = conn.cursor()
        for i in range(len(table_name_list)):
            sys.stdout.write('    [正在下载       ]    %s' % table_name_list[i] + '\r')
            url = self.server + 'hk/service/cwsj_service.php?symbol={}&start={}&end={}&type={}&unit=yuan'.format(code, table_date_list[i][-1], table_date_list[i][0], url_list[i])
            req_table = requests.get(url=url, headers=self.headers)
            table = req_table.json()
            nums = len(table)
            value_dict = {}
            for num in range(nums):
                sys.stdout.write('    [正在下载 %.2f%%]   ' % ((num + 1) / nums * 100) + '\r')
                sys.stdout.flush()
                value_dict['股票名'] = name
                value_dict['股票代码'] = code
                for (key, value) in table[i].items():
                    if key in self.table_dict[url_list[i]]:
                        value_dict[self.table_dict[url_list[i]][key]] = value
                sql1 = "\n\t\t\t\tINSERT INTO %s (`股票名`,`股票代码`,`报表日期`) VALUES ('%s','%s','%s')" % (url_list[i], value_dict['股票名'], value_dict['股票代码'], value_dict['报表日期'])
                try:
                    cursor.execute(sql1)
                    conn.commit()
                except:
                    conn.rollback()
                for (key, value) in value_dict.items():
                    if key not in ['股票名', '股票代码', '报表日期']:
                        sql2 = "\n\t\t\t\t\t\tUPDATE %s SET %s='%s' WHERE `股票名`='%s' AND `报表日期`='%s'" % (url_list[i], key, value, value_dict['股票名'], value_dict['报表日期'])
                        try:
                            cursor.execute(sql2)
                            conn.commit()
                        except:
                            conn.rollback()
                value_dict = {}
            print('    [下载完成 ')
        cursor.close()
        conn.close()
if __name__ == '__main__':
    print('*' * 100)
    print('\t\t\t\t\t财务数据下载助手\n')
    print('作者:Jack-Cui\n')
    print('About Me:\n')
    print('  知乎:https://www.zhihu.com/people/Jack--Cui')
    print('  Blog:http://blog.csdn.net/c406495762')
    print('  Gihub:https://github.com/Jack-Cherish\n')
    print('*' * 100)
    fd = FinancialData()
    code = input('请输入股票代码:')
    (name, table_name_list, table_date_list, url_list) = fd.get_informations(fd.cwnb + code + '.html')
    print('\n  %s:(%s)财务数据下载中！\n' % (name, code))
    fd.insert_tables(name, table_name_list, table_date_list, url_list)
    print('\n  %s:(%s)财务数据下载完成！' % (name, code))