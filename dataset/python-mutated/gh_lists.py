"""
gh_lists.py MILESTONE

Functions for Github API requests.
"""
import os
import re
import sys
import json
import collections
import argparse
import datetime
import time
from urllib.request import urlopen, Request, HTTPError
Issue = collections.namedtuple('Issue', ('id', 'title', 'url'))

def main():
    if False:
        return 10
    p = argparse.ArgumentParser(usage=__doc__.lstrip())
    p.add_argument('--project', default='scipy/scipy')
    p.add_argument('milestone')
    args = p.parse_args()
    getter = CachedGet('gh_cache.json', GithubGet())
    try:
        milestones = get_milestones(getter, args.project)
        if args.milestone not in milestones:
            msg = 'Milestone {0} not available. Available milestones: {1}'
            msg = msg.format(args.milestone, ', '.join(sorted(milestones)))
            p.error(msg)
        issues = get_issues(getter, args.project, args.milestone)
        issues.sort()
    finally:
        getter.save()
    prs = [x for x in issues if '/pull/' in x.url]
    issues = [x for x in issues if x not in prs]

    def print_list(title, items):
        if False:
            return 10
        print()
        print(title)
        print('-' * len(title))
        print()
        for issue in items:
            msg = '* `#{0} <{1}>`__: {2}'
            title = re.sub('\\s+', ' ', issue.title.strip())
            title = title.replace('`', '\\`').replace('*', '\\*')
            if len(title) > 60:
                remainder = re.sub('\\s.*$', '...', title[60:])
                if len(remainder) > 20:
                    remainder = title[:80] + '...'
                else:
                    title = title[:60] + remainder
            msg = msg.format(issue.id, issue.url, title)
            print(msg)
        print()
    msg = 'Issues closed for {0}'.format(args.milestone)
    print_list(msg, issues)
    msg = 'Pull requests for {0}'.format(args.milestone)
    print_list(msg, prs)
    return 0

def get_milestones(getter, project):
    if False:
        for i in range(10):
            print('nop')
    url = 'https://api.github.com/repos/{project}/milestones'.format(project=project)
    data = getter.get(url)
    milestones = {}
    for ms in data:
        milestones[ms['title']] = ms['number']
    return milestones

def get_issues(getter, project, milestone):
    if False:
        while True:
            i = 10
    milestones = get_milestones(getter, project)
    mid = milestones[milestone]
    url = 'https://api.github.com/repos/{project}/issues?milestone={mid}&state=closed&sort=created&direction=asc'
    url = url.format(project=project, mid=mid)
    data = getter.get(url)
    issues = []
    for issue_data in data:
        if 'pull' in issue_data['html_url']:
            merge_status = issue_data['pull_request']['merged_at']
            if merge_status is None:
                continue
        issues.append(Issue(issue_data['number'], issue_data['title'], issue_data['html_url']))
    return issues

class CachedGet:

    def __init__(self, filename, getter):
        if False:
            print('Hello World!')
        self._getter = getter
        self.filename = filename
        if os.path.isfile(filename):
            print('[gh_lists] using {0} as cache (remove it if you want fresh data)'.format(filename), file=sys.stderr)
            with open(filename, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def get(self, url):
        if False:
            while True:
                i = 10
        if url not in self.cache:
            data = self._getter.get_multipage(url)
            self.cache[url] = data
            return data
        else:
            print('[gh_lists] (cached):', url, file=sys.stderr, flush=True)
            return self.cache[url]

    def save(self):
        if False:
            return 10
        tmp = self.filename + '.new'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f)
        os.rename(tmp, self.filename)

class GithubGet:

    def __init__(self, auth=False):
        if False:
            while True:
                i = 10
        self.headers = {'User-Agent': 'gh_lists.py', 'Accept': 'application/vnd.github.v3+json'}
        if auth:
            self.authenticate()
        req = self.urlopen('https://api.github.com/rate_limit')
        try:
            if req.getcode() != 200:
                raise RuntimeError()
            info = json.loads(req.read().decode('utf-8'))
        finally:
            req.close()
        self.ratelimit_remaining = int(info['rate']['remaining'])
        self.ratelimit_reset = float(info['rate']['reset'])

    def authenticate(self):
        if False:
            return 10
        print("Input a Github API access token.\nPersonal tokens can be created at https://github.com/settings/tokens\nThis script does not require any permissions (so don't give it any).", file=sys.stderr, flush=True)
        print('Access token: ', file=sys.stderr, end='', flush=True)
        token = input()
        self.headers['Authorization'] = 'token {0}'.format(token.strip())

    def urlopen(self, url, auth=None):
        if False:
            for i in range(10):
                print('nop')
        assert url.startswith('https://')
        req = Request(url, headers=self.headers)
        return urlopen(req, timeout=60)

    def get_multipage(self, url):
        if False:
            i = 10
            return i + 15
        data = []
        while url:
            (page_data, info, next_url) = self.get(url)
            data += page_data
            url = next_url
        return data

    def get(self, url):
        if False:
            return 10
        while True:
            while self.ratelimit_remaining == 0 and self.ratelimit_reset > time.time():
                s = self.ratelimit_reset + 5 - time.time()
                if s <= 0:
                    break
                print('[gh_lists] rate limit exceeded: waiting until {0} ({1} s remaining)'.format(datetime.datetime.fromtimestamp(self.ratelimit_reset).strftime('%Y-%m-%d %H:%M:%S'), int(s)), file=sys.stderr, flush=True)
                time.sleep(min(5 * 60, s))
            print('[gh_lists] get:', url, file=sys.stderr, flush=True)
            try:
                req = self.urlopen(url)
                try:
                    code = req.getcode()
                    info = req.info()
                    data = json.loads(req.read().decode('utf-8'))
                finally:
                    req.close()
            except HTTPError as err:
                code = err.getcode()
                info = err.info()
                data = None
            if code not in (200, 403):
                raise RuntimeError()
            next_url = None
            if 'Link' in info:
                m = re.search('<([^<>]*)>; rel="next"', info['Link'])
                if m:
                    next_url = m.group(1)
            if 'X-RateLimit-Remaining' in info:
                self.ratelimit_remaining = int(info['X-RateLimit-Remaining'])
            if 'X-RateLimit-Reset' in info:
                self.ratelimit_reset = float(info['X-RateLimit-Reset'])
            if code != 200 or data is None:
                if self.ratelimit_remaining == 0:
                    continue
                else:
                    raise RuntimeError()
            return (data, info, next_url)
if __name__ == '__main__':
    sys.exit(main())