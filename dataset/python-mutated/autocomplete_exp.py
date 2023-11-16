import schemas
from chalicelib.core import countries, events, metadata
from chalicelib.utils import ch_client
from chalicelib.utils import helper, exp_ch_helper
from chalicelib.utils.event_filter_definition import Event
TABLE = 'experimental.autocomplete'

def __get_autocomplete_table(value, project_id):
    if False:
        return 10
    autocomplete_events = [schemas.FilterType.rev_id, schemas.EventType.click, schemas.FilterType.user_device, schemas.FilterType.user_id, schemas.FilterType.user_browser, schemas.FilterType.user_os, schemas.EventType.custom, schemas.FilterType.user_country, schemas.FilterType.user_city, schemas.FilterType.user_state, schemas.EventType.location, schemas.EventType.input]
    autocomplete_events.sort()
    sub_queries = []
    c_list = []
    for e in autocomplete_events:
        if e == schemas.FilterType.user_country:
            c_list = countries.get_country_code_autocomplete(value)
            if len(c_list) > 0:
                sub_queries.append(f"(SELECT DISTINCT ON(value) '{e.value}' AS _type, value\n                                        FROM {TABLE}\n                                        WHERE project_id = %(project_id)s\n                                            AND type= '{e.value.upper()}' \n                                            AND value IN %(c_list)s)")
            continue
        sub_queries.append(f"(SELECT '{e.value}' AS _type, value\n                                FROM {TABLE}\n                                WHERE project_id = %(project_id)s\n                                    AND type= '{e.value.upper()}' \n                                    AND value ILIKE %(svalue)s\n                                ORDER BY value\n                                LIMIT 5)")
        if len(value) > 2:
            sub_queries.append(f"(SELECT '{e.value}' AS _type, value\n                                    FROM {TABLE}\n                                    WHERE project_id = %(project_id)s\n                                        AND type= '{e.value.upper()}' \n                                        AND value ILIKE %(value)s\n                                    ORDER BY value\n                                    LIMIT 5)")
    with ch_client.ClickHouseClient() as cur:
        query = ' UNION DISTINCT '.join(sub_queries) + ';'
        params = {'project_id': project_id, 'value': helper.string_to_sql_like(value), 'svalue': helper.string_to_sql_like('^' + value), 'c_list': tuple(c_list)}
        results = []
        try:
            results = cur.execute(query=query, params=params)
        except Exception as err:
            print('--------- CH AUTOCOMPLETE SEARCH QUERY EXCEPTION -----------')
            print(cur.format(query=query, params=params))
            print('--------- PARAMS -----------')
            print(params)
            print('--------- VALUE -----------')
            print(value)
            print('--------------------')
            raise err
    for r in results:
        r['type'] = r.pop('_type')
    results = helper.list_to_camel_case(results)
    return results

def __generic_query(typename, value_length=None):
    if False:
        for i in range(10):
            print('nop')
    if typename == schemas.FilterType.user_country:
        return f"SELECT DISTINCT value, type\n                    FROM {TABLE}\n                    WHERE\n                      project_id = %(project_id)s\n                      AND type='{typename.upper()}'\n                      AND value IN %(value)s\n                      ORDER BY value"
    if value_length is None or value_length > 2:
        return f"(SELECT DISTINCT value, type\n                    FROM {TABLE}\n                    WHERE\n                      project_id = %(project_id)s\n                      AND type='{typename.upper()}'\n                      AND value ILIKE %(svalue)s\n                      ORDER BY value\n                    LIMIT 5)\n                    UNION DISTINCT\n                    (SELECT DISTINCT value, type\n                    FROM {TABLE}\n                    WHERE\n                      project_id = %(project_id)s\n                      AND type='{typename.upper()}'\n                      AND value ILIKE %(value)s\n                      ORDER BY value\n                    LIMIT 5);"
    return f"SELECT DISTINCT value, type\n                FROM {TABLE}\n                WHERE\n                  project_id = %(project_id)s\n                  AND type='{typename.upper()}'\n                  AND value ILIKE %(svalue)s\n                  ORDER BY value\n                LIMIT 10;"

def __generic_autocomplete(event: Event):
    if False:
        return 10

    def f(project_id, value, key=None, source=None):
        if False:
            return 10
        with ch_client.ClickHouseClient() as cur:
            query = __generic_query(event.ui_type, value_length=len(value))
            params = {'project_id': project_id, 'value': helper.string_to_sql_like(value), 'svalue': helper.string_to_sql_like('^' + value)}
            results = cur.execute(query=query, params=params)
            return helper.list_to_camel_case(results)
    return f

def __generic_autocomplete_metas(typename):
    if False:
        while True:
            i = 10

    def f(project_id, text):
        if False:
            while True:
                i = 10
        with ch_client.ClickHouseClient() as cur:
            params = {'project_id': project_id, 'value': helper.string_to_sql_like(text), 'svalue': helper.string_to_sql_like('^' + text)}
            if typename == schemas.FilterType.user_country:
                params['value'] = tuple(countries.get_country_code_autocomplete(text))
                if len(params['value']) == 0:
                    return []
            query = __generic_query(typename, value_length=len(text))
            rows = cur.execute(query=query, params=params)
        return rows
    return f

def __pg_errors_query(source=None, value_length=None):
    if False:
        print('Hello World!')
    MAIN_TABLE = exp_ch_helper.get_main_js_errors_sessions_table()
    if value_length is None or value_length > 2:
        return f"((SELECT DISTINCT ON(message)\n                        message AS value,\n                        source,\n                        '{events.EventType.ERROR.ui_type}' AS type\n                    FROM {MAIN_TABLE}\n                    WHERE\n                      project_id = %(project_id)s\n                      AND message ILIKE %(svalue)s\n                      AND event_type = 'ERROR'\n                      {('AND source = %(source)s' if source is not None else '')}\n                    LIMIT 5)\n                    UNION DISTINCT\n                    (SELECT DISTINCT ON(name)\n                        name AS value,\n                        source,\n                        '{events.EventType.ERROR.ui_type}' AS type\n                    FROM {MAIN_TABLE}\n                    WHERE\n                      project_id = %(project_id)s\n                      AND name ILIKE %(svalue)s\n                      {('AND source = %(source)s' if source is not None else '')}\n                    LIMIT 5)\n                    UNION DISTINCT\n                    (SELECT DISTINCT ON(message)\n                        message AS value,\n                        source,\n                        '{events.EventType.ERROR.ui_type}' AS type\n                    FROM {MAIN_TABLE}\n                    WHERE\n                      project_id = %(project_id)s\n                      AND message ILIKE %(value)s\n                      {('AND source = %(source)s' if source is not None else '')}\n                    LIMIT 5)\n                    UNION DISTINCT\n                    (SELECT DISTINCT ON(name)\n                        name AS value,\n                        source,\n                        '{events.EventType.ERROR.ui_type}' AS type\n                    FROM {MAIN_TABLE}\n                    WHERE\n                      project_id = %(project_id)s\n                      AND name ILIKE %(value)s\n                      {('AND source = %(source)s' if source is not None else '')}\n                    LIMIT 5));"
    return f"((SELECT DISTINCT ON(message)\n                    message AS value,\n                    source,\n                    '{events.EventType.ERROR.ui_type}' AS type\n                FROM {MAIN_TABLE}\n                WHERE\n                  project_id = %(project_id)s\n                  AND message ILIKE %(svalue)s\n                  {('AND source = %(source)s' if source is not None else '')}\n                LIMIT 5)\n                UNION DISTINCT\n                (SELECT DISTINCT ON(name)\n                    name AS value,\n                    source,\n                    '{events.EventType.ERROR.ui_type}' AS type\n                FROM {MAIN_TABLE}\n                WHERE\n                  project_id = %(project_id)s\n                  AND name ILIKE %(svalue)s\n                  {('AND source = %(source)s' if source is not None else '')}\n                LIMIT 5));"

def __search_errors(project_id, value, key=None, source=None):
    if False:
        i = 10
        return i + 15
    with ch_client.ClickHouseClient() as cur:
        query = cur.format(__pg_errors_query(source, value_length=len(value)), {'project_id': project_id, 'value': helper.string_to_sql_like(value), 'svalue': helper.string_to_sql_like('^' + value), 'source': source})
        results = cur.execute(query)
    return helper.list_to_camel_case(results)

def __search_errors_ios(project_id, value, key=None, source=None):
    if False:
        i = 10
        return i + 15
    return []

def __search_metadata(project_id, value, key=None, source=None):
    if False:
        i = 10
        return i + 15
    meta_keys = metadata.get(project_id=project_id)
    meta_keys = {m['key']: m['index'] for m in meta_keys}
    if len(meta_keys) == 0 or (key is not None and key not in meta_keys.keys()):
        return []
    sub_from = []
    if key is not None:
        meta_keys = {key: meta_keys[key]}
    for k in meta_keys.keys():
        colname = metadata.index_to_colname(meta_keys[k])
        if len(value) > 2:
            sub_from.append(f"((SELECT DISTINCT ON ({colname}) {colname} AS value, '{k}' AS key \n                                FROM {exp_ch_helper.get_main_sessions_table()} \n                                WHERE project_id = %(project_id)s \n                                AND {colname} ILIKE %(svalue)s LIMIT 5)\n                                UNION DISTINCT\n                                (SELECT DISTINCT ON ({colname}) {colname} AS value, '{k}' AS key \n                                FROM {exp_ch_helper.get_main_sessions_table()} \n                                WHERE project_id = %(project_id)s \n                                AND {colname} ILIKE %(value)s LIMIT 5))\n                                ")
        else:
            sub_from.append(f"(SELECT DISTINCT ON ({colname}) {colname} AS value, '{k}' AS key \n                                FROM {exp_ch_helper.get_main_sessions_table()} \n                                WHERE project_id = %(project_id)s\n                                AND {colname} ILIKE %(svalue)s LIMIT 5)")
    with ch_client.ClickHouseClient() as cur:
        query = cur.format(f"SELECT key, value, 'METADATA' AS TYPE\n                                FROM({' UNION ALL '.join(sub_from)}) AS all_metas\n                                LIMIT 5;", {'project_id': project_id, 'value': helper.string_to_sql_like(value), 'svalue': helper.string_to_sql_like('^' + value)})
        results = cur.execute(query)
    return helper.list_to_camel_case(results)