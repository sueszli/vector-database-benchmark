from datetime import datetime, timedelta
from urllib.request import urlopen
import os

def scrape_station(station):
    if False:
        print('Hello World!')
    '\n    This function scrapes the weather data web pages from wunderground.com\n    for the station you provide it.\n\n    You can look up your city\'s weather station by performing a search for\n    it on wunderground.com then clicking on the "History" section.\n    The 4-letter name of the station will appear on that page.\n    '
    current_date = datetime(year=2014, month=7, day=1)
    end_date = datetime(year=2015, month=7, day=1)
    os.mkdir(station)
    lookup_URL = 'http://www.wunderground.com/history/airport/{}/{}/{}/{}/DailyHistory.html'
    while current_date != end_date:
        if current_date.day == 1:
            print(current_date)
        formatted_lookup_URL = lookup_URL.format(station, current_date.year, current_date.month, current_date.day)
        html = urlopen(formatted_lookup_URL).read().decode('utf-8')
        out_file_name = '{}/{}-{}-{}.html'.format(station, current_date.year, current_date.month, current_date.day)
        with open(out_file_name, 'w') as out_file:
            out_file.write(html)
        current_date += timedelta(days=1)
for station in ['KCLT', 'KCQT', 'KHOU', 'KIND', 'KJAX', 'KMDW', 'KNYC', 'KPHL', 'KPHX', 'KSEA']:
    scrape_station(station)