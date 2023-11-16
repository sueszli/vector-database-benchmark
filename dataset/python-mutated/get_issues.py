import json
from pathlib import Path
import requests
issues_url = 'https://api.github.com/repos/pytest-dev/pytest/issues'

def get_issues():
    if False:
        return 10
    issues = []
    url = issues_url
    while 1:
        get_data = {'state': 'all'}
        r = requests.get(url, params=get_data)
        data = r.json()
        if r.status_code == 403:
            print(data['message'])
            exit(1)
        issues.extend(data)
        links = requests.utils.parse_header_links(r.headers['Link'])
        another_page = False
        for link in links:
            if link['rel'] == 'next':
                url = link['url']
                another_page = True
        if not another_page:
            return issues

def main(args):
    if False:
        while True:
            i = 10
    cachefile = Path(args.cache)
    if not cachefile.exists() or args.refresh:
        issues = get_issues()
        cachefile.write_text(json.dumps(issues), 'utf-8')
    else:
        issues = json.loads(cachefile.read_text('utf-8'))
    open_issues = [x for x in issues if x['state'] == 'open']
    open_issues.sort(key=lambda x: x['number'])
    report(open_issues)

def _get_kind(issue):
    if False:
        return 10
    labels = [label['name'] for label in issue['labels']]
    for key in ('bug', 'enhancement', 'proposal'):
        if key in labels:
            return key
    return 'issue'

def report(issues):
    if False:
        return 10
    for issue in issues:
        title = issue['title']
        kind = _get_kind(issue)
        status = issue['state']
        number = issue['number']
        link = 'https://github.com/pytest-dev/pytest/issues/%s/' % number
        print('----')
        print(status, kind, link)
        print(title)
    print('\n\nFound %s open issues' % len(issues))
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('process bitbucket issues')
    parser.add_argument('--refresh', action='store_true', help='invalidate cache, refresh issues')
    parser.add_argument('--cache', action='store', default='issues.json', help='cache file')
    args = parser.parse_args()
    main(args)