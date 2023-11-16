import os
import re
import sys
import httpx
PR_checkTemplate = ['Paddle']
REPO_TEMPLATE = {'Paddle': '### PR types(.*[^\\s].*)### PR changes(.*[^\\s].*)### Description(.*[^\\s].*)'}

def re_rule(body, CHECK_TEMPLATE):
    if False:
        while True:
            i = 10
    PR_RE = re.compile(CHECK_TEMPLATE, re.DOTALL)
    result = PR_RE.search(body)
    return result

def parameter_accuracy(body):
    if False:
        print('Hello World!')
    PR_dic = {}
    PR_types = ['New features', 'Bug fixes', 'Function optimization', 'Performance optimization', 'Breaking changes', 'Others']
    PR_changes = ['OPs', 'APIs', 'Docs', 'Others']
    body = re.sub('\r\n', '', body)
    type_end = body.find('### PR changes')
    changes_end = body.find('### Description')
    PR_dic['PR types'] = body[len('### PR types'):type_end]
    PR_dic['PR changes'] = body[type_end + 14:changes_end]
    message = ''
    for key in PR_dic:
        test_list = PR_types if key == 'PR types' else PR_changes
        test_list_lower = [l.lower() for l in test_list]
        value = PR_dic[key].strip().split(',')
        single_mess = ''
        if len(value) == 1 and value[0] == '':
            message += f'{key} should be in {test_list}. but now is None.'
        else:
            for i in value:
                i = i.strip().lower()
                if i not in test_list_lower:
                    single_mess += '%s.' % i
            if len(single_mess) != 0:
                message += '{} should be in {}. but now is [{}].'.format(key, test_list, single_mess)
    return message

def checkComments(url):
    if False:
        while True:
            i = 10
    headers = {'Authorization': 'token ' + GITHUB_API_TOKEN}
    response = httpx.get(url, headers=headers, timeout=None, follow_redirects=True).json()
    return response

def checkPRTemplate(repo, body, CHECK_TEMPLATE):
    if False:
        i = 10
        return i + 15
    "\n    Check if PR's description meet the standard of template\n    Args:\n        body: PR's Body.\n        CHECK_TEMPLATE: check template str.\n    Returns:\n        res: True or False\n    "
    res = False
    note = '<!-- Demo: https://github.com/PaddlePaddle/Paddle/pull/24810 -->\\r\\n|<!-- One of \\[ New features \\| Bug fixes \\| Function optimization \\| Performance optimization \\| Breaking changes \\| Others \\] -->|<!-- One of \\[ OPs \\| APIs \\| Docs \\| Others \\] -->|<!-- Describe what you’ve done -->'
    if body is None:
        body = ''
    body = re.sub(note, '', body)
    result = re_rule(body, CHECK_TEMPLATE)
    message = ''
    if len(CHECK_TEMPLATE) == 0 and len(body) == 0:
        res = False
    elif result is not None:
        message = parameter_accuracy(body)
        res = True if message == '' else False
    elif result is None:
        res = False
        message = parameter_accuracy(body)
    return (res, message)

def pull_request_event_template(event, repo, *args, **kwargs):
    if False:
        return 10
    pr_effect_repos = PR_checkTemplate
    pr_num = event['number']
    url = event['comments_url']
    BODY = event['body']
    sha = event['head']['sha']
    title = event['title']
    pr_user = event['user']['login']
    print(f'receive data : pr_num: {pr_num}, title: {title}, user: {pr_user}')
    if repo in pr_effect_repos:
        CHECK_TEMPLATE = REPO_TEMPLATE[repo]
        global check_pr_template
        global check_pr_template_message
        (check_pr_template, check_pr_template_message) = checkPRTemplate(repo, BODY, CHECK_TEMPLATE)
        print(f'check_pr_template: {check_pr_template} pr: {pr_num}')
        if check_pr_template is False:
            print('ERROR MESSAGE:', check_pr_template_message)
            sys.exit(7)
        else:
            sys.exit(0)

def get_a_pull(pull_id):
    if False:
        while True:
            i = 10
    url = 'https://api.github.com/repos/PaddlePaddle/Paddle/pulls/ ' + str(pull_id)
    payload = {}
    headers = {'Authorization': 'token ' + GITHUB_API_TOKEN, 'Accept': 'application/vnd.github+json'}
    response = httpx.request('GET', url, headers=headers, data=payload, follow_redirects=True)
    return response.json()

def main(org, repo, pull_id):
    if False:
        i = 10
        return i + 15
    pull_info = get_a_pull(pull_id)
    pull_request_event_template(pull_info, repo)
if __name__ == '__main__':
    AGILE_PULL_ID = os.getenv('AGILE_PULL_ID')
    GITHUB_API_TOKEN = os.getenv('GITHUB_API_TOKEN')
    main('PaddlePaddle', 'Paddle', AGILE_PULL_ID)