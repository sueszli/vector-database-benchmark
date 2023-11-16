import argparse
import sys
from github import Github
import os
import re

def print_pulls(repo_name, title, pulls):
    if False:
        while True:
            i = 10
    if len(pulls) > 0:
        print('**{}:**'.format(title))
        print()
        for (pull, commit) in pulls:
            url = 'https://github.com/{}/pull/{}'.format(repo_name, pull.number)
            print('- {} [#{}]({}) ({})'.format(pull.title, pull.number, url, commit.author.login))
        print()

def generate_changelog(repo, repo_name, tag1, tag2):
    if False:
        return 10
    print(f'Fetching list of commits between {tag1} and {tag2}', file=sys.stderr)
    comparison = repo.compare(tag1, tag2)
    print('Fetching pull requests', file=sys.stderr)
    unique_pulls = []
    all_pulls = []
    for commit in comparison.commits:
        pulls = commit.get_pulls()
        for pull in pulls:
            if pull.number not in unique_pulls:
                unique_pulls.append(pull.number)
                all_pulls.append((pull, commit))
    breaking = []
    bugs = []
    docs = []
    enhancements = []
    performance = []
    print('Categorizing pull requests', file=sys.stderr)
    for (pull, commit) in all_pulls:
        cc_type = ''
        cc_scope = ''
        cc_breaking = ''
        parts = re.findall('^([a-z]+)(\\([a-z]+\\))?(!)?:', pull.title)
        if len(parts) == 1:
            parts_tuple = parts[0]
            cc_type = parts_tuple[0]
            cc_scope = parts_tuple[1]
            cc_breaking = parts_tuple[2] == '!'
        labels = [label.name for label in pull.labels]
        if 'api change' in labels or cc_breaking:
            breaking.append((pull, commit))
        elif 'bug' in labels or cc_type == 'fix':
            bugs.append((pull, commit))
        elif 'performance' in labels or cc_type == 'perf':
            performance.append((pull, commit))
        elif 'enhancement' in labels or cc_type == 'feat':
            enhancements.append((pull, commit))
        elif 'documentation' in labels or cc_type == 'docs':
            docs.append((pull, commit))
    print('Generating changelog content', file=sys.stderr)
    print_pulls(repo_name, 'Breaking changes', breaking)
    print_pulls(repo_name, 'Performance related', performance)
    print_pulls(repo_name, 'Implemented enhancements', enhancements)
    print_pulls(repo_name, 'Fixed bugs', bugs)
    print_pulls(repo_name, 'Documentation updates', docs)
    print_pulls(repo_name, 'Merged pull requests', all_pulls)

def cli(args=None):
    if False:
        return 10
    'Process command line arguments.'
    if not args:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('project', help='The project name e.g. apache/arrow-datafusion')
    parser.add_argument('tag1', help='The previous release tag')
    parser.add_argument('tag2', help='The current release tag')
    args = parser.parse_args()
    token = os.getenv('GITHUB_TOKEN')
    g = Github(token)
    repo = g.get_repo(args.project)
    generate_changelog(repo, args.project, args.tag1, args.tag2)
if __name__ == '__main__':
    cli()