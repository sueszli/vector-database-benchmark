import os
import shutil
import sys
from typing import List
from dotenv import load_dotenv
from agbenchmark.execute_sub_process import execute_subprocess
load_dotenv()
helicone_graphql_logs = os.getenv('HELICONE_GRAPHQL_LOGS')
HELICONE_GRAPHQL_LOGS = helicone_graphql_logs.lower() == 'true' if helicone_graphql_logs else False

def run_agent(task: str, timeout: int) -> None:
    if False:
        return 10
    print(f'Running agbenchmark/benchmarks.py with timeout {timeout}')
    command = [sys.executable, '-m', 'agbenchmark_config.benchmarks', str(task)]
    execute_subprocess(command, timeout)

def get_list_of_file_paths(challenge_dir_path: str, artifact_folder_name: str) -> List[str]:
    if False:
        i = 10
        return i + 15
    source_dir = os.path.join(challenge_dir_path, artifact_folder_name)
    if not os.path.exists(source_dir):
        return []
    return [os.path.join(source_dir, file_name) for file_name in os.listdir(source_dir)]

def copy_artifacts_into_temp_folder(workspace: str | dict[str, str], artifact_folder_name: str, challenge_dir_path: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    file_paths = get_list_of_file_paths(challenge_dir_path, artifact_folder_name)
    for file_path in file_paths:
        if os.path.isfile(file_path):
            shutil.copy(file_path, workspace)