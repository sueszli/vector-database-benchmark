import json
import re
from dateutil.parser import parse as dateparse
from twisted.python import log
from buildbot.util import bytes2unicode
from buildbot.www.hooks.base import BaseHookHandler

class GitoriousHandler(BaseHookHandler):

    def getChanges(self, request):
        if False:
            print('Hello World!')
        payload = json.loads(bytes2unicode(request.args[b'payload'][0]))
        user = payload['repository']['owner']['name']
        repo = payload['repository']['name']
        repo_url = payload['repository']['url']
        project = payload['project']['name']
        changes = self.process_change(payload, user, repo, repo_url, project)
        log.msg(f'Received {len(changes)} changes from gitorious')
        return (changes, 'git')

    def process_change(self, payload, user, repo, repo_url, project):
        if False:
            return 10
        changes = []
        newrev = payload['after']
        branch = payload['ref']
        if re.match('^0*$', newrev):
            log.msg(f"Branch `{branch}' deleted, ignoring")
            return []
        else:
            for commit in payload['commits']:
                files = []
                when_timestamp = dateparse(commit['timestamp'])
                log.msg(f"New revision: {commit['id'][:8]}")
                changes.append({'author': f"{commit['author']['name']} <{commit['author']['email']}>", 'files': files, 'comments': commit['message'], 'revision': commit['id'], 'when_timestamp': when_timestamp, 'branch': branch, 'revlink': commit['url'], 'repository': repo_url, 'project': project})
        return changes
gitorious = GitoriousHandler