"""
@project: PyCharm
@file: history_data.py
@author: Shengqiang Zhang
@time: 2019/8/5 21:44
@mail: sqzhang77@gmail.com
"""
import sqlite3

def query_sqlite_db(history_db, query):
    if False:
        print('Hello World!')
    conn = sqlite3.connect(history_db)
    cursor = conn.cursor()
    select_statement = query
    cursor.execute(select_statement)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results

def get_history_data(history_file_path):
    if False:
        print('Hello World!')
    try:
        select_statement = 'SELECT urls.id, urls.url, urls.title, urls.last_visit_time, urls.visit_count, visits.visit_time, visits.from_visit, visits.transition, visits.visit_duration FROM urls, visits WHERE urls.id = visits.url;'
        result = query_sqlite_db(history_file_path, select_statement)
        result_sort = sorted(result, key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]))
        return result_sort
    except:
        return 'error'

def get_search_word(history_file_path):
    if False:
        i = 10
        return i + 15
    try:
        select_statement = 'SELECT keyword_search_terms.url_id, keyword_search_terms.term, urls.url, urls.last_visit_time from keyword_search_terms LEFT JOIN urls on keyword_search_terms.url_id=urls.id;'
        result = query_sqlite_db(history_file_path, select_statement)
        result_sort = sorted(result, key=lambda x: x[0])
        return result_sort
    except:
        return 'error'