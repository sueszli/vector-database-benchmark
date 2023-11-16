import datetime
import re
'\nRun the file to update the user_events.json and user_events_some_invalid.json files with more recent timestamp\n'

def update_events_timestamp(json_file):
    if False:
        for i in range(10):
            print('nop')
    request_time = datetime.datetime.now() - datetime.timedelta(days=1)
    day = request_time.date().strftime('%Y-%m-%d')
    print(day)
    with open(json_file, 'r') as file:
        filedata = file.read()
    filedata = re.sub('"eventTime":"([0-9]{4})-([0-9]{2})-([0-9]{2})', '"eventTime":"' + day, filedata, flags=re.M)
    with open(json_file, 'w') as file:
        file.write(filedata)
    print(f'The {json_file} is updated')
if __name__ == '__main__':
    update_events_timestamp('../resources/user_events.json')
    update_events_timestamp('../resources/user_events_some_invalid.json')