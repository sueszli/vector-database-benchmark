from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from urllib.request import urlopen

def parse_station(station):
    if False:
        while True:
            i = 10
    '\n    This function parses the web pages downloaded from wunderground.com\n    into a flat CSV file for the station you provide it.\np\n    Make sure to run the wunderground scraper first so you have the web\n    pages downloaded.\n    '
    current_date = datetime(year=2014, month=7, day=1)
    end_date = datetime(year=2015, month=7, day=1)
    with open('{}.csv'.format(station), 'w') as out_file:
        out_file.write('date,actual_mean_temp,actual_min_temp,actual_max_temp,average_min_temp,average_max_temp,record_min_temp,record_max_temp,record_min_temp_year,record_max_temp_year,actual_precipitation,average_precipitation,record_precipitation\n')
        while current_date != end_date:
            try_again = False
            with open('{}/{}-{}-{}.html'.format(station, current_date.year, current_date.month, current_date.day)) as in_file:
                soup = BeautifulSoup(in_file.read(), 'html.parser')
                weather_data_rows = soup.find(id='historyTable').find_all('tr')
                weather_data = []
                for i in range(len(weather_data_rows)):
                    soup1 = weather_data_rows[i]
                    weather_data.append(soup1.find_all('span', class_='wx-value'))
                    weather_data = [x for x in weather_data if x != []]
                if len(weather_data[4]) < 2:
                    weather_data[4].append(None)
                    weather_data[4].append(None)
                weather_data_units = soup.find(id='historyTable').find_all('td')
                try:
                    actual_mean_temp = weather_data[0][0].text
                    actual_max_temp = weather_data[1][0].text
                    average_max_temp = weather_data[1][1].text
                    if weather_data[1][2]:
                        record_max_temp = weather_data[1][2].text
                    actual_min_temp = weather_data[2][0].text
                    average_min_temp = weather_data[2][1].text
                    record_min_temp = weather_data[2][2].text
                    record_max_temp_year = weather_data_units[9].text.split('(')[-1].strip(')')
                    record_min_temp_year = weather_data_units[13].text.split('(')[-1].strip(')')
                    actual_precipitation = weather_data[4][0].text
                    if actual_precipitation == 'T':
                        actual_precipitation = '0.0'
                    if weather_data[4][1]:
                        average_precipitation = weather_data[4][1].text
                    else:
                        average_precipitation = None
                    if weather_data[4][2]:
                        record_precipitation = weather_data[4][2].text
                    else:
                        record_precipitation = None
                    if record_max_temp_year == '-1' or record_min_temp_year == '-1' or int(record_max_temp) < max(int(actual_max_temp), int(average_max_temp)) or (int(record_min_temp) > min(int(actual_min_temp), int(average_min_temp))) or ((record_precipitation is not None or average_precipitation is not None) and (float(actual_precipitation) > float(record_precipitation) or float(average_precipitation) > float(record_precipitation))):
                        raise Exception
                    out_file.write('{}-{}-{},'.format(current_date.year, current_date.month, current_date.day))
                    out_file.write(','.join([actual_mean_temp, actual_min_temp, actual_max_temp, average_min_temp, average_max_temp, record_min_temp, record_max_temp, record_min_temp_year, record_max_temp_year, actual_precipitation]))
                    if average_precipitation:
                        out_file.write(',{}'.format(average_precipitation))
                    if record_precipitation:
                        out_file.write(',{}'.format(record_precipitation))
                    out_file.write('\n')
                    current_date += timedelta(days=1)
                except:
                    try_again = True
            if try_again:
                print('Error with date {}'.format(current_date))
                lookup_URL = 'http://www.wunderground.com/history/airport/{}/{}/{}/{}/DailyHistory.html'
                formatted_lookup_URL = lookup_URL.format(station, current_date.year, current_date.month, current_date.day)
                html = urlopen(formatted_lookup_URL).read().decode('utf-8')
                out_file_name = '{}/{}-{}-{}.html'.format(station, current_date.year, current_date.month, current_date.day)
                with open(out_file_name, 'w') as out_file:
                    out_file.write(html)
for station in ['KCLT', 'KCQT', 'KHOU', 'KIND', 'KJAX', 'KMDW', 'KNYC', 'KPHL', 'KPHX', 'KSEA', 'KSAF']:
    parse_station(station)