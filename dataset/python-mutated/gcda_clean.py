""" usage: gcda_clean.py pull_id. """
import os
import sys
import time
from github import Github

def get_pull(pull_id):
    if False:
        for i in range(10):
            print('nop')
    'Get pull.\n\n    Args:\n        pull_id (int): Pull id.\n\n    Returns:\n        github.PullRequest.PullRequest\n    '
    token = os.getenv('GITHUB_API_TOKEN')
    github = Github(token, timeout=60)
    idx = 1
    while idx < 4:
        try:
            repo = github.get_repo('PaddlePaddle/Paddle')
        except Exception as e:
            print(e)
            print(f'get_repo error, retry {idx} times after {idx * 10} secs.')
        else:
            break
        idx += 1
        time.sleep(idx * 10)
    pull = repo.get_pull(pull_id)
    return pull

def get_files(pull_id):
    if False:
        i = 10
        return i + 15
    'Get files.\n\n    Args:\n        pull_id (int): Pull id.\n\n    Returns:\n       iterable: The generator will yield every filename.\n    '
    pull = get_pull(pull_id)
    for file in pull.get_files():
        yield file.filename

def clean(pull_id):
    if False:
        while True:
            i = 10
    'Clean.\n\n    Args:\n        pull_id (int): Pull id.\n\n    Returns:\n        None.\n    '
    changed = []
    for file in get_files(pull_id):
        changed.append(f'/paddle/build/{file}.gcda')
    for (parent, dirs, files) in os.walk('/paddle/build/'):
        for gcda in files:
            if gcda.endswith('.gcda'):
                trimmed = parent
                trimmed_tmp = []
                for p in trimmed.split('/'):
                    if p.endswith('.dir') or p.endswith('CMakeFiles'):
                        continue
                    trimmed_tmp.append(p)
                trimmed = '/'.join(trimmed_tmp)
                if os.path.join(trimmed, gcda) not in changed:
                    gcda = os.path.join(parent, gcda)
                    os.remove(gcda)
if __name__ == '__main__':
    pull_id = sys.argv[1]
    pull_id = int(pull_id)
    clean(pull_id)