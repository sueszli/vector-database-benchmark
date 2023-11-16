import glob
import json
import os
from typing import Dict, List, Optional, Union
import pandas as pd
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from pydantic import BaseModel, Field

class Metrics(BaseModel):
    difficulty: str
    success: bool
    success_percent: float = Field(..., alias='success_%')
    run_time: Optional[str] = None
    fail_reason: Optional[str] = None
    attempted: Optional[bool] = None

class MetricsOverall(BaseModel):
    run_time: str
    highest_difficulty: str
    percentage: Optional[float] = None

class Test(BaseModel):
    data_path: str
    is_regression: bool
    answer: str
    description: str
    metrics: Metrics
    category: List[str]
    task: Optional[str] = None
    reached_cutoff: Optional[bool] = None

class SuiteTest(BaseModel):
    data_path: str
    metrics: MetricsOverall
    tests: Dict[str, Test]
    category: Optional[List[str]] = None
    task: Optional[str] = None
    reached_cutoff: Optional[bool] = None

class Report(BaseModel):
    command: str
    completion_time: str
    benchmark_start_time: str
    metrics: MetricsOverall
    tests: Dict[str, Union[Test, SuiteTest]]
    config: Dict[str, str | dict[str, str]]

def get_reports():
    if False:
        while True:
            i = 10
    report_data = []
    current_dir = os.getcwd()
    if current_dir.endswith('reports'):
        reports_dir = '/'
    else:
        reports_dir = 'reports'
    for agent_name in os.listdir(reports_dir):
        if agent_name is None:
            continue
        agent_dir = os.path.join(reports_dir, agent_name)
        if os.path.isdir(agent_dir):
            run_dirs = glob.glob(os.path.join(agent_dir, '*'))
            report_files = [os.path.join(run_dir, 'report.json') for run_dir in run_dirs if os.path.isdir(run_dir)]
            for report_file in report_files:
                if os.path.isfile(report_file):
                    with open(report_file, 'r') as f:
                        json_data = json.load(f)
                        print(f'Processing {report_file}')
                        report = Report.parse_obj(json_data)
                        for (test_name, test_data) in report.tests.items():
                            test_json = {'agent': agent_name.lower(), 'benchmark_start_time': report.benchmark_start_time}
                            if isinstance(test_data, SuiteTest):
                                if test_data.category:
                                    test_json['challenge'] = test_name
                                    test_json['attempted'] = test_data.tests[list(test_data.tests.keys())[0]].metrics.attempted
                                    test_json['categories'] = ', '.join(test_data.category)
                                    test_json['task'] = test_data.task
                                    test_json['success'] = test_data.metrics.percentage
                                    test_json['difficulty'] = test_data.metrics.highest_difficulty
                                    test_json['success_%'] = test_data.metrics.percentage
                                    test_json['run_time'] = test_data.metrics.run_time
                                    test_json['is_regression'] = test_data.tests[list(test_data.tests.keys())[0]].is_regression
                                else:
                                    for (suite_test_name, suite_data) in test_data.tests.items():
                                        test_json['challenge'] = suite_test_name
                                        test_json['attempted'] = suite_data.metrics.attempted
                                        test_json['categories'] = ', '.join(suite_data.category)
                                        test_json['task'] = suite_data.task
                                        test_json['success'] = 100.0 if suite_data.metrics.success else 0
                                        test_json['difficulty'] = suite_data.metrics.difficulty
                                        test_json['success_%'] = suite_data.metrics.success_percentage
                                        test_json['run_time'] = suite_data.metrics.run_time
                                        test_json['is_regression'] = suite_data.is_regression
                            else:
                                test_json['challenge'] = test_name
                                test_json['attempted'] = test_data.metrics.attempted
                                test_json['categories'] = ', '.join(test_data.category)
                                test_json['task'] = test_data.task
                                test_json['success'] = 100.0 if test_data.metrics.success else 0
                                test_json['difficulty'] = test_data.metrics.difficulty
                                test_json['success_%'] = test_data.metrics.success_percentage
                                test_json['run_time'] = test_data.metrics.run_time
                                test_json['is_regression'] = test_data.is_regression
                            report_data.append(test_json)
    return pd.DataFrame(report_data)

def get_helicone_data():
    if False:
        return 10
    helicone_api_key = os.getenv('HELICONE_API_KEY')
    url = 'https://www.helicone.ai/api/graphql'
    transport = AIOHTTPTransport(url=url, headers={'authorization': f'Bearer {helicone_api_key}'})
    client = Client(transport=transport, fetch_schema_from_transport=True)
    SIZE = 250
    i = 0
    data = []
    print('Fetching data from Helicone')
    while True:
        query = gql('\n            query ExampleQuery($limit: Int, $offset: Int){\n                heliconeRequest(\n                    limit: $limit\n                    offset: $offset\n                ) {\n                    costUSD\n                    prompt\n                    properties{\n                        name\n                        value\n                    }\n                    \n                    requestBody\n                    response\n                    createdAt\n\n                }\n\n                }\n        ')
        print(f'Fetching {i * SIZE} to {(i + 1) * SIZE} records')
        try:
            result = client.execute(query, variable_values={'limit': SIZE, 'offset': i * SIZE})
        except Exception as e:
            print(f'Error occurred: {e}')
            result = None
        i += 1
        if result:
            for item in result['heliconeRequest']:
                properties = {prop['name']: prop['value'] for prop in item['properties']}
                data.append({'createdAt': item['createdAt'], 'agent': properties.get('agent'), 'costUSD': item['costUSD'], 'job_id': properties.get('job_id'), 'challenge': properties.get('challenge'), 'benchmark_start_time': properties.get('benchmark_start_time'), 'prompt': item['prompt'], 'response': item['response'], 'model': item['requestBody'].get('model'), 'request': item['requestBody'].get('messages')})
        if not result or len(result['heliconeRequest']) == 0:
            print('No more results')
            break
    df = pd.DataFrame(data)
    df = df.dropna(subset=['agent'])
    df['agent'] = df['agent'].str.lower()
    return df
if os.path.exists('raw_reports.pkl') and os.path.exists('raw_helicone.pkl'):
    reports_df = pd.read_pickle('raw_reports.pkl')
    helicone_df = pd.read_pickle('raw_helicone.pkl')
else:
    reports_df = get_reports()
    reports_df.to_pickle('raw_reports.pkl')
    helicone_df = get_helicone_data()
    helicone_df.to_pickle('raw_helicone.pkl')

def try_formats(date_str):
    if False:
        return 10
    formats = ['%Y-%m-%d-%H:%M', '%Y-%m-%dT%H:%M:%S%z']
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            pass
    return None
helicone_df['benchmark_start_time'] = pd.to_datetime(helicone_df['benchmark_start_time'].apply(try_formats), utc=True)
helicone_df = helicone_df.dropna(subset=['benchmark_start_time'])
helicone_df['createdAt'] = pd.to_datetime(helicone_df['createdAt'], unit='ms', origin='unix')
reports_df['benchmark_start_time'] = pd.to_datetime(reports_df['benchmark_start_time'].apply(try_formats), utc=True)
reports_df = reports_df.dropna(subset=['benchmark_start_time'])
assert pd.api.types.is_datetime64_any_dtype(helicone_df['benchmark_start_time']), 'benchmark_start_time in helicone_df is not datetime'
assert pd.api.types.is_datetime64_any_dtype(reports_df['benchmark_start_time']), 'benchmark_start_time in reports_df is not datetime'
reports_df['report_time'] = reports_df['benchmark_start_time']
df = pd.merge(helicone_df, reports_df, on=['benchmark_start_time', 'agent', 'challenge'], how='inner')
df.to_pickle('df.pkl')
print(df.info())
print('Data saved to df.pkl')
print("To load the data use: df = pd.read_pickle('df.pkl')")