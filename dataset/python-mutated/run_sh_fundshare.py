"""
@author:xda
@file:run_sh_fundshare.py
@time:2021/01/24
"""
from fund_share_crawl import SHFundShare
import fire

def main(kind, date='now'):
    if False:
        i = 10
        return i + 15
    '\n    LOF 20210101\n    ETF 2021-01-01\n    :param kind:\n    :param date:\n    :return:\n    '
    app = SHFundShare(first_use=False, kind=kind, date=date)
    app.run()
if __name__ == '__main__':
    '\n    --kind=ETF --date=now #\n    '
    fire.Fire(main)