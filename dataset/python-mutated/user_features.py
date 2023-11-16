from utils.pg_client import PostgresClient
from decouple import config
from utils.df_utils import _process_pg_response
import numpy as np

def get_training_database(projectId, max_timestamp=None, favorites=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Gets training database using projectId, max_timestamp [optional] and favorites (if true adds favorites)\n    Params:\n        projectId: project id of all sessions to be selected.\n        max_timestamp: max timestamp that a not seen session can have in order to be considered not interesting.\n        favorites: True to use favorite sessions as interesting sessions reference.\n    Output: Tuple (Set of features, set of labels, dict of indexes of each project_id, session_id, user_id in the set)\n    '
    args = {'projectId': projectId, 'max_timestamp': max_timestamp, 'limit': 20}
    with PostgresClient() as conn:
        x1 = signals_features(conn, **args)
        if favorites:
            x2 = user_favorite_sessions(args['projectId'], conn)
        if max_timestamp is not None:
            x3 = user_not_seen_sessions(args['projectId'], args['limit'], conn)
    X_project_ids = dict()
    X_users_ids = dict()
    X_sessions_ids = dict()
    _X = list()
    _Y = list()
    _process_pg_response(x1, _X, _Y, X_project_ids, X_users_ids, X_sessions_ids, label=None)
    if favorites:
        _process_pg_response(x2, _X, _Y, X_project_ids, X_users_ids, X_sessions_ids, label=1)
    if max_timestamp:
        _process_pg_response(x3, _X, _Y, X_project_ids, X_users_ids, X_sessions_ids, label=0)
    return (np.array(_X), np.array(_Y), {'project_id': X_project_ids, 'user_id': X_users_ids, 'session_id': X_sessions_ids})

def signals_features(conn, **kwargs):
    if False:
        print('Hello World!')
    '\n    Selects features from frontend_signals table and mark as interesting given the following conditions:\n        * If number of events is greater than events_threshold (default=10). (env value)\n        * If session has been replayed more than once.\n    '
    assert 'projectId' in kwargs.keys(), 'projectId should be provided in kwargs'
    projectId = kwargs['projectId']
    events_threshold = config('events_threshold', default=10, cast=int)
    query = conn.mogrify("SELECT T.project_id,\n                                   T.session_id,\n                                   T.user_id,\n                                   T2.viewer_id,\n                                   T.events_count,\n                                   T.errors_count,\n                                   T.duration,\n                                   T.country,\n                                   T.issue_score,\n                                   T.device_type,\n                                   T2.interesting as train_label\n                            FROM (SELECT project_id,\n                                         user_id                                                            as viewer_id,\n                                         session_id,\n                                         count(CASE WHEN source = 'replay' THEN 1 END) > 1 OR COUNT(1) > %(events_threshold)s as interesting\n                                  FROM frontend_signals\n                                  WHERE project_id = %(projectId)s\n                                    AND session_id is not null\n                                  GROUP BY project_id, viewer_id, session_id) as T2\n                                     INNER JOIN (SELECT project_id,\n                                                        session_id,\n                                                        user_id          as viewer_id,\n                                                        user_id,\n                                                        events_count,\n                                                        errors_count,\n                                                        duration,\n                                                        user_country     as country,\n                                                        issue_score,\n                                                        user_device_type as device_type\n                                                 FROM sessions\n                                                 WHERE project_id = %(projectId)s\n                                                   AND duration IS NOT NULL) as T\n                                                USING (session_id);", {'projectId': projectId, 'events_threshold': events_threshold})
    conn.execute(query)
    res = conn.fetchall()
    return res

def user_favorite_sessions(projectId, conn):
    if False:
        return 10
    '\n    Selects features from user_favorite_sessions table.\n    '
    query = 'SELECT project_id,\n                       session_id,\n                       T1.user_id,\n                       events_count,\n                       errors_count,\n                       duration,\n                       user_country     as country,\n                       issue_score,\n                       user_device_type as device_type,\n                       T2.user_id       AS viewer_id\n                FROM sessions AS T1\n                         INNER JOIN user_favorite_sessions as T2\n                                    USING (session_id)\n                WHERE project_id = %(projectId)s;'
    conn.execute(conn.mogrify(query, {'projectId': projectId}))
    res = conn.fetchall()
    return res

def user_not_seen_sessions(projectId, limit, conn):
    if False:
        print('Hello World!')
    '\n    Selects features from user_viewed_sessions table.\n    '
    query = 'SELECT project_id, session_id, user_id, viewer_id, events_count, errors_count, duration, user_country as country, issue_score, user_device_type as device_type\nFROM (\n         (SELECT sessions.*\n         FROM sessions LEFT JOIN user_viewed_sessions USING(session_id)\n         WHERE project_id = %(projectId)s  \n            AND duration IS NOT NULL\n            AND user_viewed_sessions.session_id ISNULL\n         LIMIT %(limit)s) AS T1\n             LEFT JOIN\n         (SELECT user_id as viewer_id\n         FROM users\n         WHERE tenant_id = (SELECT tenant_id FROM projects WHERE project_id = %(projectId)s)) AS T2 ON true\n     )'
    conn.execute(conn.mogrify(query, {'projectId': projectId, 'limit': limit}))
    res = conn.fetchall()
    return res