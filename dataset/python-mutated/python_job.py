import logging
import os
import shutil
import sys
import tempfile
from pyflink.table import EnvironmentSettings, TableEnvironment
from pyflink.table.expressions import col, call, lit

def word_count():
    if False:
        print('Hello World!')
    content = 'line Licensed to the Apache Software Foundation ASF under one line or more contributor license agreements See the NOTICE file line distributed with this work for additional information line regarding copyright ownership The ASF licenses this file to you under the Apache License Version the License you may not use this file except in compliance with the License'
    t_env = TableEnvironment.create(EnvironmentSettings.in_batch_mode())
    config_key = sys.argv[1]
    config_value = sys.argv[2]
    t_env.get_config().set(config_key, config_value)
    tmp_dir = tempfile.gettempdir()
    result_path = tmp_dir + '/result'
    if os.path.exists(result_path):
        try:
            if os.path.isfile(result_path):
                os.remove(result_path)
            else:
                shutil.rmtree(result_path)
        except OSError as e:
            logging.error('Error removing directory: %s - %s.', e.filename, e.strerror)
    logging.info('Results directory: %s', result_path)
    sink_ddl = "\n        create table Results(\n            word VARCHAR,\n            `count` BIGINT,\n            `count_java` BIGINT\n        ) with (\n            'connector.type' = 'filesystem',\n            'format.type' = 'csv',\n            'connector.path' = '{}'\n        )\n        ".format(result_path)
    t_env.execute_sql(sink_ddl)
    t_env.execute_sql("create temporary system function add_one as 'add_one.add_one' language python")
    t_env.register_java_function('add_one_java', 'org.apache.flink.python.tests.util.AddOne')
    elements = [(word, 0) for word in content.split(' ')]
    t = t_env.from_elements(elements, ['word', 'count'])
    t.select(t.word, call('add_one', t.count).alias('count'), call('add_one_java', t.count).alias('count_java')).group_by(t.word).select(t.word, col('count').count.alias('count'), col('count_java').count.alias('count_java')).execute_insert('Results')
if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    word_count()