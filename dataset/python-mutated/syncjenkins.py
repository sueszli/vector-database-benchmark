import os
import psycopg2
import requests
import socket
import sys
import time
from datetime import datetime, timedelta
from xml.etree import ElementTree
host = os.environ['DB_HOST']
port = os.environ['DB_PORT']
dbname = os.environ['DB_DBNAME']
dbusername = os.environ['DB_DBUSERNAME']
dbpassword = os.environ['DB_DBPWD']
jenkinsBuildsTableName = 'jenkins_builds'
jenkinsJobsCreateTableQuery = f'\ncreate table {jenkinsBuildsTableName} (\njob_name varchar NOT NULL,\nbuild_id integer NOT NULL,\nbuild_url varchar,\nbuild_result varchar,\nbuild_timestamp TIMESTAMP,\nbuild_builtOn varchar,\nbuild_duration BIGINT,\nbuild_estimatedDuration BIGINT,\nbuild_fullDisplayName varchar,\ntiming_blockedDurationMillis BIGINT,\ntiming_buildableDurationMillis BIGINT,\ntiming_buildingDurationMillis BIGINT,\ntiming_executingTimeMillis BIGINT,\ntiming_queuingDurationMillis BIGINT,\ntiming_totalDurationMillis BIGINT,\ntiming_waitingDurationMillis BIGINT,\nprimary key(job_name, build_id)\n)\n'

def fetchJobs():
    if False:
        for i in range(10):
            print('nop')
    url = 'https://ci-beam.apache.org/api/json?tree=jobs[name,url,lastCompletedBuild[id]]&depth=1'
    r = requests.get(url)
    jobs = r.json()['jobs']
    result = map(lambda x: (x['name'], int(x['lastCompletedBuild']['id']) if x['lastCompletedBuild'] is not None else -1, x['url']), jobs)
    return result

def initConnection():
    if False:
        while True:
            i = 10
    conn = None
    while not conn:
        try:
            conn = psycopg2.connect(f"dbname='{dbname}' user='{dbusername}' host='{host}' port='{port}' password='{dbpassword}'")
        except:
            print('Failed to connect to DB; retrying in 1 minute')
            time.sleep(60)
    return conn

def tableExists(cursor, tableName):
    if False:
        i = 10
        return i + 15
    cursor.execute(f"select * from information_schema.tables where table_name='{tableName}';")
    return bool(cursor.rowcount)

def initDbTablesIfNeeded():
    if False:
        while True:
            i = 10
    connection = initConnection()
    cursor = connection.cursor()
    buildsTableExists = tableExists(cursor, jenkinsBuildsTableName)
    print('Builds table exists', buildsTableExists)
    if not buildsTableExists:
        cursor.execute(jenkinsJobsCreateTableQuery)
        if not bool(cursor.rowcount):
            raise Exception(f'Failed to create table {jenkinsBuildsTableName}')
    cursor.close()
    connection.commit()
    connection.close()

def fetchLastSyncTimestamp(cursor):
    if False:
        while True:
            i = 10
    fetchQuery = f'\n  select job_name, max(build_id)\n  from {jenkinsBuildsTableName}\n  group by job_name\n  '
    cursor.execute(fetchQuery)
    return dict(cursor.fetchall())

def fetchBuildsForJob(jobUrl):
    if False:
        i = 10
        return i + 15
    durFields = 'blockedDurationMillis,buildableDurationMillis,buildingDurationMillis,executingTimeMillis,queuingDurationMillis,totalDurationMillis,waitingDurationMillis'
    fields = f'result,timestamp,id,url,builtOn,building,duration,estimatedDuration,fullDisplayName,actions[{durFields}]'
    url = f'{jobUrl}api/json?depth=1&tree=builds[{fields}]'
    r = requests.get(url)
    return r.json()['builds']

def buildRowValuesArray(jobName, build):
    if False:
        for i in range(10):
            print('nop')
    timings = next((x for x in build['actions'] if '_class' in x and x['_class'] == 'jenkins.metrics.impl.TimeInQueueAction'), None)
    values = [jobName, int(build['id']), build['url'], build['result'], datetime.fromtimestamp(build['timestamp'] / 1000), build['builtOn'], build['duration'], build['estimatedDuration'], build['fullDisplayName'], timings['blockedDurationMillis'] if timings is not None else -1, timings['buildableDurationMillis'] if timings is not None else -1, timings['buildingDurationMillis'] if timings is not None else -1, timings['executingTimeMillis'] if timings is not None else -1, timings['queuingDurationMillis'] if timings is not None else -1, timings['totalDurationMillis'] if timings is not None else -1, timings['waitingDurationMillis'] if timings is not None else -1]
    return values

def insertRow(cursor, rowValues):
    if False:
        i = 10
        return i + 15
    cursor.execute(f'insert into {jenkinsBuildsTableName} values (%s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', rowValues)

def fetchNewData():
    if False:
        for i in range(10):
            print('nop')
    connection = initConnection()
    cursor = connection.cursor()
    syncedJobs = fetchLastSyncTimestamp(cursor)
    cursor.close()
    connection.close()
    newJobs = fetchJobs()
    for (newJobName, newJobLastBuildId, newJobUrl) in newJobs:
        syncedJobId = syncedJobs[newJobName] if newJobName in syncedJobs else -1
        if newJobLastBuildId > syncedJobId:
            builds = fetchBuildsForJob(newJobUrl)
            builds = [x for x in builds if int(x['id']) > syncedJobId]
            connection = initConnection()
            cursor = connection.cursor()
            for build in builds:
                if build['building']:
                    continue
                rowValues = buildRowValuesArray(newJobName, build)
                print('inserting', newJobName, build['id'])
                insertRow(cursor, rowValues)
            cursor.close()
            connection.commit()
            connection.close()

def probeJenkinsIsUp():
    if False:
        return 10
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('ci-beam.apache.org', 443))
    return True if result == 0 else False
if __name__ == '__main__':
    print('Started.')
    print('Checking if DB needs to be initialized.')
    sys.stdout.flush()
    initDbTablesIfNeeded()
    print('Start jobs fetching loop.')
    sys.stdout.flush()
    while True:
        if not probeJenkinsIsUp():
            print('Jenkins is unavailable, skipping fetching data.')
            continue
        else:
            fetchNewData()
            print('Fetched data.')
        print('Sleeping for 5 min.')
        sys.stdout.flush()
        time.sleep(5 * 60)
    print('Done.')