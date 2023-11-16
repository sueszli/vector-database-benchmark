"""
Topic: 分组迭代
Desc : 
"""
from operator import itemgetter
from itertools import groupby

def group_iter():
    if False:
        for i in range(10):
            print('nop')
    rows = [{'address': '5412 N CLARK', 'date': '07/01/2012'}, {'address': '5148 N CLARK', 'date': '07/04/2012'}, {'address': '5800 E 58TH', 'date': '07/02/2012'}, {'address': '2122 N CLARK', 'date': '07/03/2012'}, {'address': '5645 N RAVENSWOOD', 'date': '07/02/2012'}, {'address': '1060 W ADDISON', 'date': '07/02/2012'}, {'address': '4801 N BROADWAY', 'date': '07/01/2012'}, {'address': '1039 W GRANVILLE', 'date': '07/04/2012'}]
    rows.sort(key=itemgetter('date'))
    for (date, items) in groupby(rows, key=itemgetter('date')):
        print(date)
        for i in items:
            print(' ', i)
    from collections import defaultdict
    rows_by_date = defaultdict(list)
    for row in rows:
        rows_by_date[row['date']].append(row)
if __name__ == '__main__':
    group_iter()