import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from airbyte_cdk.logger import AirbyteLogger
from firebolt.async_db import Connection as AsyncConnection
from firebolt.async_db import connect as async_connect
from firebolt.client import DEFAULT_API_URL
from firebolt.client.auth import UsernamePassword
from firebolt.db import Connection, connect

def parse_config(config: json, logger: AirbyteLogger) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    '\n    Convert dict of config values to firebolt.db.Connection arguments\n\n    :param config: json-compatible dict of settings\n    :param logger: AirbyteLogger instance to print logs.\n\n    :return: dictionary of firebolt.db.Connection-compatible kwargs\n    '
    connection_args = {'database': config['database'], 'auth': UsernamePassword(config['username'], config['password']), 'api_endpoint': config.get('host', DEFAULT_API_URL), 'account_name': config.get('account')}
    engine = config.get('engine')
    if engine:
        if '.' in engine:
            connection_args['engine_url'] = engine
        else:
            connection_args['engine_name'] = engine
    else:
        logger.info('Engine parameter was not provided. Connecting to the default engine.')
    return connection_args

def establish_connection(config: json, logger: AirbyteLogger) -> Connection:
    if False:
        return 10
    '\n    Creates a connection to Firebolt database using the parameters provided.\n\n    :param config: Json object containing db credentials.\n    :param logger: AirbyteLogger instance to print logs.\n\n    :return: PEP-249 compliant database Connection object.\n    '
    logger.debug('Connecting to Firebolt.')
    connection = connect(**parse_config(config, logger))
    logger.debug('Connection to Firebolt established.')
    return connection

async def establish_async_connection(config: json, logger: AirbyteLogger) -> AsyncConnection:
    """
    Creates an async connection to Firebolt database using the parameters provided.
    This connection can be used for parallel operations.

    :param config: Json object containing db credentials.
    :param logger: AirbyteLogger instance to print logs.

    :return: PEP-249 compliant database Connection object.
    """
    logger.debug('Connecting to Firebolt.')
    connection = await async_connect(**parse_config(config, logger))
    logger.debug('Connection to Firebolt established.')
    return connection

def get_table_structure(connection: Connection) -> Dict[str, List[Tuple]]:
    if False:
        return 10
    '\n    Get columns and their types for all the tables and views in the database.\n\n    :param connection: Connection object connected to a database\n\n    :return: Dictionary containing column list of each table\n    '
    column_mapping = defaultdict(list)
    cursor = connection.cursor()
    cursor.execute("SELECT table_name, column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name NOT IN (SELECT table_name FROM information_schema.tables WHERE table_type IN ('EXTERNAL', 'CATALOG'))")
    for (t_name, c_name, c_type, nullable) in cursor.fetchall():
        column_mapping[t_name].append((c_name, c_type, nullable))
    cursor.close()
    return column_mapping