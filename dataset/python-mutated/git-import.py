"""Clones a git repo the parameters.

Script used by the pod in charge of pulling from git, exit codes:
- 1 -> GitCloneFailed
- 2 -> ProjectWithSameNameExists
- 3 -> ProjectNotDiscoveredByWebServer
- 4 -> NoAccessRightsOrRepoDoesNotExists
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from typing import Optional
import requests

def _git_clone_project(task_uuid: str, repo_url: str, project_name: str) -> Optional[str]:
    if False:
        while True:
            i = 10
    try:
        git_command = f'git clone {repo_url}'
        if project_name is not None and project_name:
            git_command += f' "{project_name}"'
        print(f'Running {git_command}.')
        tmp_path = f'/tmp/{task_uuid}/'
        os.mkdir(tmp_path)
        result = subprocess.run(shlex.split(git_command), cwd=tmp_path, env={**os.environ, **{'GIT_TERMINAL_PROMPT': '0'}}, capture_output=True)
        if result.returncode != 0:
            if result.stderr is not None and 'correct access rights' in result.stderr.decode().lower():
                sys.exit(4)
            sys.exit(1)
        inferred_project_name = os.listdir(tmp_path)[0]
        from_path = os.path.join(tmp_path, inferred_project_name)
        exit_code = os.system(f'mv "{from_path}" /userdir/projects/')
        if exit_code != 0:
            sys.exit(2)
        projects_gid = os.stat('/userdir/projects').st_gid
        os.system('chown -R :%s "%s"' % (projects_gid, os.path.join('/userdir/projects', inferred_project_name)))
        for _ in range(5):
            resp = requests.get('http://orchest-webserver/async/projects')
            if resp.status_code != 200:
                time.sleep(1)
                continue
            project = [proj for proj in resp.json() if proj['path'] == inferred_project_name]
            if project or not os.path.exists(f'/userdir/projects/{inferred_project_name}'):
                return project[0]['uuid'] if project else None
            time.sleep(1)
        sys.exit(3)
    finally:
        os.system(f'rm -rf "{tmp_path}"')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-uuid', type=str, required=True)
    parser.add_argument('--repo-url', type=str, required=True)
    parser.add_argument('--project-name', type=str, default=None, required=False)
    args = parser.parse_args()
    project_uuid = _git_clone_project(task_uuid=args.task_uuid, repo_url=args.repo_url, project_name=args.project_name)
    print()
    print(project_uuid)