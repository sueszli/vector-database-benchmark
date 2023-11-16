"""Check if the PR has a news item.

Put a warning comment if it doesn't.
"""
import os
from fnmatch import fnmatch
from github import Github, PullRequest

def get_added_files(pr: PullRequest.PullRequest):
    if False:
        while True:
            i = 10
    print(pr, pr.number)
    for file in pr.get_files():
        if file.status == 'added':
            yield file.filename

def check_news_file(pr):
    if False:
        print('Hello World!')
    return any(map(lambda file_name: fnmatch(file_name, 'news/*.rst'), get_added_files(pr)))

def get_pr_number():
    if False:
        for i in range(10):
            print('nop')
    number = os.environ['PR_NUMBER']
    if not number:
        raise Exception(f'Pull request number is not found `PR_NUMBER={number}')
    return int(number)

def get_old_comment(pr: PullRequest.PullRequest):
    if False:
        while True:
            i = 10
    for comment in pr.get_issue_comments():
        if 'github-actions' in comment.user.login and 'No news item is found' in comment.body:
            return comment

def main():
    if False:
        while True:
            i = 10
    gh = Github(os.environ['GITHUB_TOKEN'])
    repo = gh.get_repo(os.environ['GITHUB_REPOSITORY'])
    pr = repo.get_pull(get_pr_number())
    has_news_added = check_news_file(pr)
    old_comment = get_old_comment(pr)
    if old_comment:
        print('Found an existing comment from bot')
        if has_news_added:
            print('Delete warning from bot, since news items is added.')
            old_comment.delete()
    elif not has_news_added:
        print('No news item found')
        pr.create_issue_comment('**Warning!** No news item is found for this PR.\nIf this is an user facing change/feature/fix, please add a news item by copying the format from `news/TEMPLATE.rst`.\n')
if __name__ == '__main__':
    main()