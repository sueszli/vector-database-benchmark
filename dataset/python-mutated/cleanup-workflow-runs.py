import argparse
import os
import re
import subprocess
from datetime import datetime, timedelta
from github import Github
DAYS_TO_KEEP_ORPHANED_JOBS = 90
"\n\nThis script is intended to be run in conjuction with identify-dormant-workflows.py to keep GH actions clean.\n\nThe basic workflow is \n\nidentify-dormant-workflows.py notifies of dormant workflows (workflows that have no runs newer than DAYS_TO_KEEP_ORPHANED_JOBS days) daily -> \nmanually notifies infra team via slack ->\ninfra team checks with stakeholders to ensure dormant jobs can be deleted and then cleans up workflow runs manually ->\ncleanup-workflows.py deletes old workflow runs (again older than DAYS_TO_KEEP_ORPHANED_JOBS) that have no associated workflow\n\nWe need to clean up the runs because even if a workflow is deleted, the runs linger in the UI. \nWe don't want to delete workflow runs newer than 90 days on GH actions, even if the workflow doesn't exist.\nit's possible that people might test things off the master branch and we don't want to delete their recent runs\n\n"
parser = argparse.ArgumentParser()
parser.add_argument('--pat', '-p', help='Set github personal access token')
parser.add_argument('--delete', '-d', action='store', nargs='*', help='By default, the script will only print runs that will be deleted. Pass --delete to actually delete them')

def main():
    if False:
        return 10
    args = parser.parse_args()
    token = None
    if args.pat:
        token = args.pat
    else:
        token = os.getenv('GITHUB_TOKEN')
    if not token:
        raise Exception('Github personal access token not provided via args and not available in GITHUB_TOKEN variable')
    g = Github(token)
    git_url = subprocess.run(['git', 'config', '--get', 'remote.origin.url'], check=True, capture_output=True)
    git_url_regex = re.compile('(?:git@|https://)github\\.com[:/](.*?)(\\.git|$)')
    re_match = git_url_regex.match(git_url.stdout.decode('utf-8'))
    repo = g.get_repo(re_match.group(1))
    workflows = repo.get_workflows()
    runs_to_delete = []
    for workflow in workflows:
        if not os.path.exists(workflow.path):
            runs = workflow.get_runs()
            for run in runs:
                if run.updated_at > datetime.now() - timedelta(days=DAYS_TO_KEEP_ORPHANED_JOBS):
                    break
                if args.delete is not None:
                    print('Deleting run id ' + str(run.id))
                    run._requester.requestJson('DELETE', run.url)
                else:
                    runs_to_delete.append((workflow.name, run.id, run.created_at.strftime('%m/%d/%Y, %H:%M:%S')))
    if args.delete is None:
        print('[DRY RUN] A total of ' + str(len(runs_to_delete)) + ' runs would be deleted: ')
        for run in runs_to_delete:
            print(run)
if __name__ == '__main__':
    main()