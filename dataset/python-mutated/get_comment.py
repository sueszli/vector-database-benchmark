import os
import requests

def get_versions(versions_file):
    if False:
        print('Hello World!')
    'Get the versions of the packages used in the linter job.\n\n    Parameters\n    ----------\n    versions_file : str\n        The path to the file that contains the versions of the packages.\n\n    Returns\n    -------\n    versions : dict\n        A dictionary with the versions of the packages.\n    '
    with open('versions.txt', 'r') as f:
        return dict((line.strip().split('=') for line in f))

def get_step_message(log, start, end, title, message, details):
    if False:
        return 10
    'Get the message for a specific test.\n\n    Parameters\n    ----------\n    log : str\n        The log of the linting job.\n\n    start : str\n        The string that marks the start of the test.\n\n    end : str\n        The string that marks the end of the test.\n\n    title : str\n        The title for this section.\n\n    message : str\n        The message to be added at the beginning of the section.\n\n    details : bool\n        Whether to add the details of each step.\n\n    Returns\n    -------\n    message : str\n        The message to be added to the comment.\n    '
    if end not in log:
        return ''
    res = '-----------------------------------------------\n' + f'### {title}\n\n' + message + '\n\n'
    if details:
        res += '<details>\n\n```\n' + log[log.find(start) + len(start) + 1:log.find(end) - 1] + '\n```\n\n</details>\n\n'
    return res

def get_message(log_file, repo, pr_number, sha, run_id, details, versions):
    if False:
        while True:
            i = 10
    with open(log_file, 'r') as f:
        log = f.read()
    sub_text = f'\n\n<sub> _Generated for commit: [{sha[:7]}](https://github.com/{repo}/pull/{pr_number}/commits/{sha}). Link to the linter CI: [here](https://github.com/{repo}/actions/runs/{run_id})_ </sub>'
    if '### Linting completed ###' not in log:
        return '## ❌ Linting issues\n\nThere was an issue running the linter job. Please update with `upstream/main` ([link](https://scikit-learn.org/dev/developers/contributing.html#how-to-contribute)) and push the changes. If you already have done that, please send an empty commit with `git commit --allow-empty` and push the changes to trigger the CI.\n\n' + sub_text
    message = ''
    message += get_step_message(log, start='### Running black ###', end='Problems detected by black', title='`black`', message=f"`black` detected issues. Please run `black .` locally and push the changes. Here you can see the detected issues. Note that running black might also fix some of the issues which might be detected by `ruff`. Note that the installed `black` version is `black={versions['black']}`.", details=details)
    message += get_step_message(log, start='### Running ruff ###', end='Problems detected by ruff', title='`ruff`', message=f"`ruff` detected issues. Please run `ruff --fix --show-source .` locally, fix the remaining issues, and push the changes. Here you can see the detected issues. Note that the installed `ruff` version is `ruff={versions['ruff']}`.", details=details)
    message += get_step_message(log, start='### Running mypy ###', end='Problems detected by mypy', title='`mypy`', message=f"`mypy` detected issues. Please fix them locally and push the changes. Here you can see the detected issues. Note that the installed `mypy` version is `mypy={versions['mypy']}`.", details=details)
    message += get_step_message(log, start='### Running cython-lint ###', end='Problems detected by cython-lint', title='`cython-lint`', message=f"`cython-lint` detected issues. Please fix them locally and push the changes. Here you can see the detected issues. Note that the installed `cython-lint` version is `cython-lint={versions['cython-lint']}`.", details=details)
    message += get_step_message(log, start='### Checking for bad deprecation order ###', end='Problems detected by deprecation order check', title='Deprecation Order', message='Deprecation order check detected issues. Please fix them locally and push the changes. Here you can see the detected issues.', details=details)
    message += get_step_message(log, start='### Checking for default doctest directives ###', end='Problems detected by doctest directive check', title='Doctest Directives', message='doctest directive check detected issues. Please fix them locally and push the changes. Here you can see the detected issues.', details=details)
    message += get_step_message(log, start='### Checking for joblib imports ###', end='Problems detected by joblib import check', title='Joblib Imports', message='`joblib` import check detected issues. Please fix them locally and push the changes. Here you can see the detected issues.', details=details)
    if not message:
        return '## ✔️ Linting Passed\nAll linting checks passed. Your pull request is in excellent shape! ☀️' + sub_text
    if not details:
        branch_not_updated = "_Merging with `upstream/main` might fix / improve the issues if you haven't done that since 21.06.2023._\n\n"
    else:
        branch_not_updated = ''
    message = '## ❌ Linting issues\n\n' + branch_not_updated + "This PR is introducing linting issues. Here's a summary of the issues. " + 'Note that you can avoid having linting issues by enabling `pre-commit` ' + 'hooks. Instructions to enable them can be found [here](' + 'https://scikit-learn.org/dev/developers/contributing.html#how-to-contribute)' + '.\n\n' + 'You can see the details of the linting issues under the `lint` job [here]' + f'(https://github.com/{repo}/actions/runs/{run_id})\n\n' + message + sub_text
    return message

def get_headers(token):
    if False:
        i = 10
        return i + 15
    'Get the headers for the GitHub API.'
    return {'Accept': 'application/vnd.github+json', 'Authorization': f'Bearer {token}', 'X-GitHub-Api-Version': '2022-11-28'}

def find_lint_bot_comments(repo, token, pr_number):
    if False:
        while True:
            i = 10
    'Get the comment from the linting bot.'
    response = requests.get(f'https://api.github.com/repos/{repo}/issues/{pr_number}/comments', headers=get_headers(token))
    response.raise_for_status()
    all_comments = response.json()
    failed_comment = '❌ Linting issues'
    success_comment = '✔️ Linting Passed'
    comments = [comment for comment in all_comments if comment['user']['login'] == 'github-actions[bot]' and (failed_comment in comment['body'] or success_comment in comment['body'])]
    if len(all_comments) > 25 and (not comments):
        raise RuntimeError('Comment not found in the first 30 comments.')
    return comments[0] if comments else None

def create_or_update_comment(comment, message, repo, pr_number, token):
    if False:
        while True:
            i = 10
    'Create a new comment or update existing one.'
    if comment is not None:
        print('updating existing comment')
        response = requests.patch(f"https://api.github.com/repos/{repo}/issues/comments/{comment['id']}", headers=get_headers(token), json={'body': message})
    else:
        print('creating new comment')
        response = requests.post(f'https://api.github.com/repos/{repo}/issues/{pr_number}/comments', headers=get_headers(token), json={'body': message})
    response.raise_for_status()
if __name__ == '__main__':
    repo = os.environ['GITHUB_REPOSITORY']
    token = os.environ['GITHUB_TOKEN']
    pr_number = os.environ['PR_NUMBER']
    sha = os.environ['BRANCH_SHA']
    log_file = os.environ['LOG_FILE']
    run_id = os.environ['RUN_ID']
    versions_file = os.environ['VERSIONS_FILE']
    versions = get_versions(versions_file)
    if not repo or not token or (not pr_number) or (not log_file) or (not run_id):
        raise ValueError('One of the following environment variables is not set: GITHUB_REPOSITORY, GITHUB_TOKEN, PR_NUMBER, LOG_FILE, RUN_ID')
    try:
        comment = find_lint_bot_comments(repo, token, pr_number)
    except RuntimeError:
        print('Comment not found in the first 30 comments. Skipping!')
        exit(0)
    try:
        message = get_message(log_file, repo=repo, pr_number=pr_number, sha=sha, run_id=run_id, details=True, versions=versions)
        create_or_update_comment(comment=comment, message=message, repo=repo, pr_number=pr_number, token=token)
        print(message)
    except requests.HTTPError:
        message = get_message(log_file, repo=repo, pr_number=pr_number, sha=sha, run_id=run_id, details=False, versions=versions)
        create_or_update_comment(comment=comment, message=message, repo=repo, pr_number=pr_number, token=token)
        print(message)