import csv
csvfile = '/tmp/blah.csv'

def get_file():
    if False:
        return 10
    return csvfile
csv.writer(csvfile, delimiter=',', quotechar='"')
csv.writer(get_file(), delimiter=',', quotechar='"')
csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)