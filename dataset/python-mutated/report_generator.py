import argparse
from itertools import product
import logging
import os
from influxdb import InfluxDBClient
import jinja2
from prettytable import PrettyTable
from dashboards_parser import guess_dashboard_by_measurement
INFLUXDB_USER = os.getenv('INFLUXDB_USER')
INFLUXDB_USER_PASSWORD = os.getenv('INFLUXDB_USER_PASSWORD')
WORKING_SPACE = os.getenv('GITHUB_WORKSPACE', os.getenv('WORKSPACE', ''))
if 'GITHUB_WORKSPACE' in os.environ:
    path_prefix = ''
else:
    path_prefix = 'src/'
PERF_DASHBOARDS = os.path.join(WORKING_SPACE, path_prefix + '.test-infra/metrics/grafana/dashboards/perftests_metrics/')
TABLE_FIELD_NAMES = ['Measurement', 'Metric', 'Runner', 'Mean previous week', 'Mean last week', 'Diff %', 'Dashboard']
QUERY_RUNTIME = 'SELECT mean("value") AS "mean_value" \n  FROM \n    "{database}"."{retention_policy}"."{measurement}"\n  WHERE \n      time > (now()- 2w) \n    AND \n      time < now() \n  GROUP BY time(1w), "metric" FILL(none);'
QUERY_RUNTIME_MS = 'SELECT mean("runtimeMs") AS "mean_value" \n  FROM \n    "{database}"."{retention_policy}"."{measurement}"\n  WHERE \n      time > (now()- 2w) \n    AND \n      time < now() \n  GROUP BY time(1w), "runner" FILL(none);'

def parse_arguments():
    if False:
        i = 10
        return i + 15
    '\n  Gets all necessary data.\n  Return: influx_host, influx_port, influx_db\n  '
    parser = argparse.ArgumentParser(description='Script for generating Beam Metrics Report.')
    parser.add_argument('--influx-host', required=True)
    parser.add_argument('--influx-port', required=True)
    parser.add_argument('--influx-db', required=True)
    parser.add_argument('--output-file', required=True)
    args = parser.parse_args()
    influx_host = args.influx_host
    influx_port = args.influx_port
    influx_db = args.influx_db
    output_file = args.output_file
    return (influx_host, influx_port, influx_db, output_file)

def get_retention_policies_names(client, database):
    if False:
        print('Hello World!')
    return (i.get('name') for i in client.get_list_retention_policies(database=database))

def get_measurements_names(client):
    if False:
        i = 10
        return i + 15
    return (i.get('name') for i in client.get_list_measurements())

def calc_diff(prev, curr):
    if False:
        while True:
            i = 10
    'Returns percentage difference between two values.'
    return (curr - prev) / prev * 100.0 if prev != 0 else float('inf') * abs(curr) / curr if curr != 0 else 0.0

def _get_query_runtime_data(client, bind_params):
    if False:
        i = 10
        return i + 15
    'Returns data for measurements with runtime, write_time or read_time metrics'
    data = []
    result = client.query(QUERY_RUNTIME.format(**bind_params))
    for i in result.items():
        measurement = i[0][0]
        metric = i[0][1].get('metric')
        runner = '-'
        measurement_data = list(i[1])
        if all((m not in metric for m in ['runtime', 'write_time', 'read_time'])):
            continue
        if len(measurement_data) >= 2:
            previous = measurement_data[-2]['mean_value']
            current = measurement_data[-1]['mean_value']
            diff = calc_diff(previous, current)
            dashboards = ['http://metrics.beam.apache.org/d/{}'.format(dashboard.uid) for dashboard in guess_dashboard_by_measurement(measurement, PERF_DASHBOARDS, ['runtime', 'write_time', 'read_time'])]
            data.append([measurement, metric, runner, round(previous, 2), round(current, 2), round(diff, 2), dashboards])
    return data

def _get_query_runtime_ms_data(client, bind_params):
    if False:
        for i in range(10):
            print('nop')
    'Returns data for measurements with RuntimeMs metrics'
    data = []
    result = client.query(QUERY_RUNTIME_MS.format(**bind_params))
    for i in result.items():
        measurement = i[0][0]
        metric = 'RuntimeMs'
        runner = i[0][1].get('runner')
        measurement_data = list(i[1])
        if len(measurement_data) >= 2:
            previous = measurement_data[-2]['mean_value']
            current = measurement_data[-1]['mean_value']
            diff = calc_diff(previous, current)
            dashboards = ['http://metrics.beam.apache.org/d/{}'.format(dashboard.uid) for dashboard in guess_dashboard_by_measurement(measurement, PERF_DASHBOARDS, [metric])]
            data.append([measurement, metric, runner, round(previous, 2), round(current, 2), round(diff, 2), dashboards])
    return data

def get_metrics_data(client, database):
    if False:
        while True:
            i = 10
    data = []
    for (retention_policy, measurements_name) in product(get_retention_policies_names(client, database), get_measurements_names(client)):
        bind_params = {'database': database, 'measurement': measurements_name, 'retention_policy': retention_policy}
        data.extend(_get_query_runtime_data(client, bind_params))
        data.extend(_get_query_runtime_ms_data(client, bind_params))
    return [d for d in data if d]

def print_table(data):
    if False:
        for i in range(10):
            print('nop')
    table = PrettyTable()
    table.field_names = TABLE_FIELD_NAMES
    for d in data:
        table.add_row(d)
    print(table)

def generate_report(data, output_file):
    if False:
        for i in range(10):
            print('nop')
    logging.info('Generating {}'.format(output_file))
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'templates')))
    template = env.get_template('Metrics_Report.template')
    with open(output_file, 'w') as file:
        file.write(template.render(headers=TABLE_FIELD_NAMES, metrics_data=data))
    logging.info('{} saved.'.format(output_file))

def main():
    if False:
        print('Hello World!')
    (influx_host, influx_port, influx_db, output_file) = parse_arguments()
    client = InfluxDBClient(host=influx_host, port=influx_port, database=influx_db, username=INFLUXDB_USER, password=INFLUXDB_USER_PASSWORD)
    data = get_metrics_data(client, influx_db)
    print_table(data)
    generate_report(data, output_file)
if __name__ == '__main__':
    main()