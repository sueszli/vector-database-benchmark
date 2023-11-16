import base64
import json
import os
import re
from datetime import datetime, timedelta
import gspread
import pandas as pd
from dotenv import load_dotenv
from oauth2client.service_account import ServiceAccountCredentials
load_dotenv()
base64_creds = os.getenv('GDRIVE_BASE64')
if base64_creds is None:
    raise ValueError('The GDRIVE_BASE64 environment variable is not set')
creds_bytes = base64.b64decode(base64_creds)
creds_string = creds_bytes.decode('utf-8')
creds_info = json.loads(creds_string)
base_dir = 'reports'
current_dir = os.getcwd()
if current_dir.endswith('reports'):
    base_dir = '/'
else:
    base_dir = 'reports'
rows = []

def process_test(test_name: str, test_info: dict, agent_name: str, common_data: dict) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Recursive function to process test data.'
    parts = test_name.split('_', 1)
    test_suite = parts[0] if len(parts) > 1 else None
    separator = '|'
    categories = separator.join(test_info.get('category', []))
    row = {'Agent': agent_name, 'Command': common_data.get('command', ''), 'Completion Time': common_data.get('completion_time', ''), 'Benchmark Start Time': common_data.get('benchmark_start_time', ''), 'Total Run Time': common_data.get('metrics', {}).get('run_time', ''), 'Highest Difficulty': common_data.get('metrics', {}).get('highest_difficulty', ''), 'Workspace': common_data.get('config', {}).get('workspace', ''), 'Test Name': test_name, 'Data Path': test_info.get('data_path', ''), 'Is Regression': test_info.get('is_regression', ''), 'Difficulty': test_info.get('metrics', {}).get('difficulty', ''), 'Success': test_info.get('metrics', {}).get('success', ''), 'Success %': test_info.get('metrics', {}).get('success_%', ''), 'Non mock success %': test_info.get('metrics', {}).get('non_mock_success_%', ''), 'Run Time': test_info.get('metrics', {}).get('run_time', ''), 'Benchmark Git Commit Sha': common_data.get('benchmark_git_commit_sha', None), 'Agent Git Commit Sha': common_data.get('agent_git_commit_sha', None), 'Cost': test_info.get('metrics', {}).get('cost', ''), 'Attempted': test_info.get('metrics', {}).get('attempted', ''), 'Test Suite': test_suite, 'Category': categories, 'Task': test_info.get('task', ''), 'Answer': test_info.get('answer', ''), 'Description': test_info.get('description', ''), 'Fail Reason': test_info.get('metrics', {}).get('fail_reason', ''), 'Reached Cutoff': test_info.get('reached_cutoff', '')}
    rows.append(row)
    nested_tests = test_info.get('tests')
    if nested_tests:
        for (nested_test_name, nested_test_info) in nested_tests.items():
            process_test(nested_test_name, nested_test_info, agent_name, common_data)
for agent_dir in os.listdir(base_dir):
    agent_dir_path = os.path.join(base_dir, agent_dir)
    if os.path.isdir(agent_dir_path):
        for report_folder in os.listdir(agent_dir_path):
            report_folder_path = os.path.join(agent_dir_path, report_folder)
            if os.path.isdir(report_folder_path):
                report_path = os.path.join(report_folder_path, 'report.json')
                if os.path.exists(report_path):
                    with open(report_path, 'r') as f:
                        data = json.load(f)
                    benchmark_start_time = data.get('benchmark_start_time', '')
                    pattern = re.compile('\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\+00:00')
                    if not pattern.fullmatch(benchmark_start_time):
                        continue
                    benchmark_datetime = datetime.strptime(benchmark_start_time, '%Y-%m-%dT%H:%M:%S+00:00')
                    current_datetime = datetime.utcnow()
                    if current_datetime - benchmark_datetime > timedelta(days=3):
                        continue
                    for (test_name, test_info) in data['tests'].items():
                        process_test(test_name, test_info, agent_dir, data)
df = pd.DataFrame(rows)
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
client = gspread.authorize(creds)
branch_name = os.getenv('GITHUB_REF_NAME')
sheet = client.open(f'benchmark-{branch_name}')
sheet_instance = sheet.get_worksheet(0)
values = df.values.tolist()
values.insert(0, df.columns.tolist())
sheet_instance.clear()
sheet_instance.append_rows(values)