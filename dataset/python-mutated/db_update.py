import asyncio
import hashlib
import mlflow
import os
import pandas as pd
import pendulum
import sys
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from datetime import datetime, timedelta
from decouple import config
_work_dir = os.getcwd()
sys.path.insert(1, _work_dir)
from utils import pg_client
from utils.feedback import ConnectionHandler
from sqlalchemy import text
execute_interval = config('EXECUTION_INTERVAL', default=24 * 60 * 60, cast=int)
features_table_name = config('FEATURES_TABLE_NAME', default='features_table')
host = config('pg_host_ml')
port = config('pg_port_ml')
user = config('pg_user_ml')
dbname = config('pg_dbname_ml')
password = config('pg_password_ml')
tracking_uri = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}'

def get_today_feedback():
    if False:
        for i in range(10):
            print('nop')
    connection_handler = ConnectionHandler(tracking_uri)
    with connection_handler.get_live_session() as conn:
        cur = conn.connection().connection.cursor()
        query = cur.mogrify(f'SELECT * FROM recommendation_feedback WHERE insertion_time > %(time_lower_bound)s;', {'time_lower_bound': int(datetime.now().timestamp()) - execute_interval})
        conn.execute(text(query.decode('utf-8')))
        conn.commit()

def get_features_pg(ti):
    if False:
        print('Hello World!')
    os.environ['PG_POOL'] = 'true'
    asyncio.run(pg_client.init())
    sessionIds = ti.xcom_pull(key='sessionIds')
    userIds = ti.xcom_pull(key='userIds').split(',')
    with pg_client.PostgresClient() as conn:
        conn.execute("SELECT T.project_id,\n                                   T.session_id,\n                                   T2.viewer_id,\n                                   T.pages_count,\n                                   T.events_count,\n                                   T.errors_count,\n                                   T.duration,\n                                   T.country,\n                                   T.issue_score,\n                                   T.device_type,\n                                   T2.replays,\n                                   T2.network_access,\n                                   T2.storage_access,\n                                   T2.console_access,\n                                   T2.stack_access\n                            FROM (SELECT project_id,\n                                         user_id                                                            as viewer_id,\n                                         session_id,\n                                         count(CASE WHEN source = 'replay' THEN 1 END) as replays,\n                                         count(CASE WHEN source = 'network' THEN 1 END) as network_access,\n                                         count(CASE WHEN source = 'storage' THEN 1 END) as storage_access,\n                                         count(CASE WHEN source = 'console' THEN 1 END) as console_access,\n                                         count(CASE WHEN source = 'stack_events' THEN 1 END) as stack_access\n                                  FROM frontend_signals\n                                  WHERE session_id IN ({sessionIds})\n                                  GROUP BY project_id, viewer_id, session_id) as T2\n                                     INNER JOIN (SELECT project_id,\n                                                        session_id,\n                                                        user_id,\n                                                        pages_count,\n                                                        events_count,\n                                                        errors_count,\n                                                        duration,\n                                                        user_country     as country,\n                                                        issue_score,\n                                                        user_device_type as device_type\n                                                 FROM sessions\n                                                 WHERE session_id IN ({sessionIds})\n                                                   AND duration IS NOT NULL) as T\n                                                USING (session_id);".format(sessionIds=sessionIds))
        response = conn.fetchall()
    sessionIds = [int(sessId) for sessId in sessionIds.split(',')]
    df = pd.DataFrame(response)
    df2 = pd.DataFrame(zip(userIds, sessionIds), columns=['viewer_id', 'session_id'])
    base_query = f'INSERT INTO {features_table_name} (project_id, session_id, viewer_id, pages_count, events_count,\n       issues_count, duration, country, issue_score, device_type,\n       replays, network_access, storage_access, console_access,\n       stack_access) VALUES '
    count = 0
    params = {}
    for i in range(len(df)):
        viewer = df['viewer_id'].iloc[i]
        session = df['session_id'].iloc[i]
        d = df2[df2['viewer_id'] == viewer]
        x = d[d['session_id'] == session]
        if len(x) > 0:
            template = '('
            for (k, v) in x.items():
                params[f'{k}_{count}'] = v.values[0]
                template += f's({k}_{count})%'
            base_query += template + '), '
            count += 1
    base_query = base_query[:-2]
    connection_handler = ConnectionHandler(tracking_uri)
    with connection_handler.get_live_session() as conn:
        cur = conn.connection().connection.cursor()
        query = cur.mogrify(base_query, params)
        conn.execute(text(query.decode('utf-8')))
        conn.commit()
dag = DAG('Feedback_DB_FILL', default_args={'retries': 1, 'retry_delay': timedelta(minutes=3)}, start_date=pendulum.datetime(2015, 12, 1, tz='UTC'), description='My first test', schedule=timedelta(seconds=execute_interval), catchup=False)
with dag:
    dag_t_feedback = PythonOperator(task_id='Get_Feedbacks', python_callable=get_today_feedback)
    dag_features = PythonOperator(task_id='Update_DB', python_callable=get_features_pg)
    dag_t_feedback >> dag_features