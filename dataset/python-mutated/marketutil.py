__author__ = 'saeedamen'
import datetime
from datetime import timedelta
import pandas as pd

class MarketUtil(object):

    def parse_date(self, date):
        if False:
            print('Hello World!')
        if isinstance(date, str):
            date1 = datetime.datetime.utcnow()
            if date is 'midnight':
                date1 = datetime.datetime(date1.year, date1.month, date1.day, 0, 0, 0)
            elif date is 'decade':
                date1 = date1 - timedelta(days=365 * 10)
            elif date is 'year':
                date1 = date1 - timedelta(days=365)
            elif date is 'month':
                date1 = date1 - timedelta(days=30)
            elif date is 'week':
                date1 = date1 - timedelta(days=7)
            elif date is 'day':
                date1 = date1 - timedelta(days=1)
            elif date is 'hour':
                date1 = date1 - timedelta(hours=1)
            else:
                try:
                    date1 = datetime.datetime.strptime(date, '%b %d %Y %H:%M')
                except:
                    i = 0
                try:
                    date1 = datetime.datetime.strptime(date, '%d %b %Y %H:%M')
                except:
                    i = 0
                try:
                    date1 = datetime.datetime.strptime(date, '%b %d %Y')
                except:
                    i = 0
                try:
                    date1 = datetime.datetime.strptime(date, '%d %b %Y')
                except:
                    i = 0
        else:
            date1 = pd.Timestamp(date)
        return pd.Timestamp(date1)