import argparse
import os
import subprocess

def get_commit_message():
    if False:
        for i in range(10):
            print('nop')
    'Retrieve the commit message.'
    build_source_version_message = os.environ['BUILD_SOURCEVERSIONMESSAGE']
    if os.environ['BUILD_REASON'] == 'PullRequest':
        commit_id = build_source_version_message.split()[1]
        git_cmd = ['git', 'log', commit_id, '-1', '--pretty=%B']
        commit_message = subprocess.run(git_cmd, capture_output=True, text=True).stdout.strip()
    else:
        commit_message = build_source_version_message
    commit_message = commit_message.replace('##vso', '..vso')
    return commit_message

def parsed_args():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Show commit message that triggered the build in Azure DevOps pipeline')
    parser.add_argument('--only-show-message', action='store_true', default=False, help='Only print commit message. Useful for direct use in scripts rather than setting output variable of the Azure job')
    return parser.parse_args()
if __name__ == '__main__':
    args = parsed_args()
    commit_message = get_commit_message()
    if args.only_show_message:
        print(commit_message)
    else:
        print(f'##vso[task.setvariable variable=message;isOutput=true]{commit_message}')
        print(f'commit message: {commit_message}')