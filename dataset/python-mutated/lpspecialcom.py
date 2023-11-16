import scrapy
import re
from datetime import datetime
import pandas as pd
import time
from liepinSpd.items import LiepinspdItem

class LiepinSpdier(scrapy.Spider):
    name = 'liepin'
    start_urls = ['https://www.liepin.com/job/1917579081.shtml?d_sfrom=search_comp&d_ckId=d112af69ab58e7da8305520f55b31904&d_curPage=0&d_pageSize=15&d_headId=d112af69ab58e7da8305520f55b31904&d_posi=0', 'https://www.liepin.com/job/1917549017.shtml?d_sfrom=search_comp&d_ckId=1bef90aa98c2e8da734552c320527ac0&d_curPage=0&d_pageSize=15&d_headId=1bef90aa98c2e8da734552c320527ac0&d_posi=0', 'https://www.liepin.com/job/1917543155.shtml', 'https://www.liepin.com/job/1917491571.shtml?d_sfrom=search_comp&d_ckId=5874ecde43eb4bd20e75fecb2709bf85&d_curPage=0&d_pageSize=15&d_headId=5874ecde43eb4bd20e75fecb2709bf85&d_posi=0', 'https://www.liepin.com/job/1917505785.shtml?d_sfrom=search_comp&d_ckId=fe82f0f79cda01b1dd4c140ced26087c&d_curPage=0&d_pageSize=15&d_headId=fe82f0f79cda01b1dd4c140ced26087c&d_posi=0', 'https://www.liepin.com/job/1916439263.shtml?d_sfrom=search_comp&d_ckId=d3f4428da37a0cd17a6235cb4a027f1e&d_curPage=0&d_pageSize=15&d_headId=d3f4428da37a0cd17a6235cb4a027f1e&d_posi=0', 'https://www.liepin.com/job/1911157736.shtml?d_sfrom=search_comp&d_ckId=2cf44398e8273003087d5148e113ef8f&d_curPage=0&d_pageSize=15&d_headId=2cf44398e8273003087d5148e113ef8f&d_posi=0', 'https://www.liepin.com/job/1917470663.shtml?d_sfrom=search_comp&d_ckId=9087e4fc55d61d200606fb906999f728&d_curPage=0&d_pageSize=15&d_headId=9087e4fc55d61d200606fb906999f728&d_posi=0', 'https://www.liepin.com/job/1917533673.shtml?d_sfrom=search_comp&d_ckId=98408645fba7219d4d7f17f2714c96f0&d_curPage=0&d_pageSize=15&d_headId=98408645fba7219d4d7f17f2714c96f0&d_posi=0', 'https://www.liepin.com/job/1917306593.shtml?d_sfrom=search_comp&d_ckId=85f632646e2b1ad7c06f436e25fd674d&d_curPage=0&d_pageSize=15&d_headId=85f632646e2b1ad7c06f436e25fd674d&d_posi=0', 'https://www.liepin.com/job/199929552.shtml']

    def parse(self, response):
        if False:
            i = 10
            return i + 15
        text = response.text
        company_name = response.xpath('//div[@class="about-position"]//a/text()')[0].extract()
        size = re.search('公司规模：(.*?)人', text).group(1)
        city = re.search('公司地址：(.*?)<', text).group(1)
        industry = re.search('行业.*?>(.*?)<', text).group(1)
        as_of_date = datetime.now()
        item = LiepinspdItem()
        data = pd.read_csv('G:\\workspace\\y2019m01\\/first_lagou\\company300.csv', encoding='gbk')
        try:
            for i in range(len(data)):
                n = 0
                for j in data.loc[i, '股票简称']:
                    if j in company_name:
                        n += 1
                if n >= len(data.loc[i, '股票简称']) - 1:
                    item['ticker'] = data.loc[i, '股票代码']
                    print(n, item['ticker'], company_name)
        except BaseException as e:
            item['ticker'] = 'None'
            print('ticker匹配错误')
        item['as_of_date'] = as_of_date
        item['company_name'] = company_name
        item['size'] = size
        item['city'] = city
        item['industry'] = industry
        yield item