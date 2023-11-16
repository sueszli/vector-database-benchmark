""" usage: cuda_clean.py pull_id. """
import os
import sys
from github import Github

def get_pull(pull_id):
    if False:
        while True:
            i = 10
    '\n    Args:\n        pull_id (int): Pull id.\n\n    Returns:\n        github.PullRequest.PullRequest: The pull request.\n    '
    token = os.getenv('GITHUB_API_TOKEN')
    github = Github(token, timeout=60)
    repo = github.get_repo('PaddlePaddle/Paddle')
    pull = repo.get_pull(pull_id)
    return pull

def get_files(pull_id):
    if False:
        i = 10
        return i + 15
    '\n    Args:\n        pull_id (int): Pull id.\n\n    Returns:\n       iterable: The generator will yield every filename.\n    '
    pull = get_pull(pull_id)
    for file in pull.get_files():
        yield file.filename

def clean(pull_id):
    if False:
        i = 10
        return i + 15
    '\n    Args:\n        pull_id (int): Pull id.\n\n    Returns:\n        None.\n    '
    changed = []
    for file in get_files(pull_id):
        changed.append(file)
    for (parent, dirs, files) in os.walk('/paddle/build/'):
        for gcda in files:
            if gcda.endswith('.gcda'):
                file_name = gcda.replace('.gcda', '')
                dir_name_list = parent.replace('/paddle/build/', '').split('/')
                dir_name_list = dir_name_list[:-2]
                dir_name = '/'.join(dir_name_list)
                src_name = dir_name + '/' + file_name
                if src_name not in changed:
                    unused_file = parent + '/' + gcda
                    os.remove(gcda)
                else:
                    print(src_name)
if __name__ == '__main__':
    pull_id = sys.argv[1]
    pull_id = int(pull_id)
    clean(pull_id)