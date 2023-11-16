"""Tool to import browser history from other browsers."""
import argparse
import sqlite3
import sys
import os

class Error(Exception):
    """Exception for errors in this module."""

def parse():
    if False:
        return 10
    'Parse command line arguments.'
    description = 'This program is meant to extract browser history from your previous browser and import them into qutebrowser.'
    epilog = "Databases:\n\n\tqutebrowser: Is named 'history.sqlite' and can be found at your --basedir. In order to find where your basedir is you can run ':open qute:version' inside qutebrowser.\n\n\tFirefox: Is named 'places.sqlite', and can be found at your system's profile folder. Check this link for where it is located: http://kb.mozillazine.org/Profile_folder\n\n\tChrome: Is named 'History', and can be found at the respective User Data Directory. Check this link for where it islocated: https://chromium.googlesource.com/chromium/src/+/master/docs/user_data_dir.md\n\nExample: hist_importer.py -b firefox -s /Firefox/Profile/places.sqlite -d /qutebrowser/data/history.sqlite"
    parser = argparse.ArgumentParser(description=description, epilog=epilog, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-b', '--browser', dest='browser', required=True, type=str, help='Browsers: {firefox, chrome}')
    parser.add_argument('-s', '--source', dest='source', required=True, type=str, help='Source: Full path to the sqlite database file from the source browser.')
    parser.add_argument('-d', '--dest', dest='dest', required=True, type=str, help='\nDestination: Full path to the qutebrowser sqlite database')
    return parser.parse_args()

def open_db(data_base):
    if False:
        return 10
    'Open connection with database.'
    if os.path.isfile(data_base):
        return sqlite3.connect(data_base)
    raise Error('The file {} does not exist.'.format(data_base))

def extract(source, query):
    if False:
        return 10
    'Get records from source database.\n\n    Args:\n        source: File path to the source database where we want to extract the\n        data from.\n        query: The query string to be executed in order to retrieve relevant\n        attributes as (datetime, url, time) from the source database according\n        to the browser chosen.\n    '
    try:
        conn = open_db(source)
        cursor = conn.cursor()
        cursor.execute(query)
        history = cursor.fetchall()
        conn.close()
        return history
    except sqlite3.OperationalError as op_e:
        raise Error('Could not perform queries on the source database: {}'.format(op_e))

def clean(history):
    if False:
        while True:
            i = 10
    "Clean up records from source database.\n\n    Receives a list of record and sanityze them in order for them to be\n    properly imported to qutebrowser. Sanitation requires adding a 4th\n    attribute 'redirect' which is filled with '0's, and also purging all\n    records that have a NULL/None datetime attribute.\n\n    Args:\n        history: List of records (datetime, url, title) from source database.\n    "
    for (index, record) in enumerate(history):
        if record[1] is None:
            cleaned = list(record)
            cleaned[1] = ''
            history[index] = tuple(cleaned)
    nulls = [record for record in history if None in record]
    for null_record in nulls:
        history.remove(null_record)
    history = [list(record) for record in history]
    for record in history:
        record.append('0')
    return history

def insert_qb(history, dest):
    if False:
        return 10
    'Insert history into dest database.\n\n    Args:\n        history: List of records.\n        dest: File path to the destination database, where history will be\n        inserted.\n    '
    conn = open_db(dest)
    cursor = conn.cursor()
    cursor.executemany('INSERT INTO History (url,title,atime,redirect) VALUES (?,?,?,?)', history)
    cursor.execute('UPDATE CompletionMetaInfo SET value = 1 WHERE key = "force_rebuild"')
    conn.commit()
    conn.close()

def run():
    if False:
        print('Hello World!')
    'Main control flux of the script.'
    args = parse()
    browser = args.browser.lower()
    (source, dest) = (args.source, args.dest)
    query = {'firefox': 'select url,title,last_visit_date/1000000 as date from moz_places where url like "http%" or url like "ftp%" or url like "file://%"', 'chrome': 'select url,title,last_visit_time/10000000 as date from urls'}
    if browser not in query:
        raise Error('Sorry, the selected browser: "{}" is not supported.'.format(browser))
    history = extract(source, query[browser])
    history = clean(history)
    insert_qb(history, dest)

def main():
    if False:
        while True:
            i = 10
    try:
        run()
    except Error as e:
        sys.exit(str(e))
if __name__ == '__main__':
    main()