import sys
import traceback
from twisted.internet import defer
from buildbot.clients import sendchange as sendchange_client
from buildbot.util import in_reactor

@in_reactor
@defer.inlineCallbacks
def sendchange(config):
    if False:
        return 10
    encoding = config.get('encoding', 'utf8')
    who = config.get('who')
    auth = config.get('auth')
    master = config.get('master')
    branch = config.get('branch')
    category = config.get('category')
    revision = config.get('revision')
    properties = config.get('properties', {})
    repository = config.get('repository', '')
    vc = config.get('vc', None)
    project = config.get('project', '')
    revlink = config.get('revlink', '')
    when = config.get('when')
    comments = config.get('comments')
    files = config.get('files', ())
    codebase = config.get('codebase', None)
    s = sendchange_client.Sender(master, auth, encoding=encoding)
    try:
        yield s.send(branch, revision, comments, files, who=who, category=category, when=when, properties=properties, repository=repository, vc=vc, project=project, revlink=revlink, codebase=codebase)
    except Exception:
        print('change not sent:')
        traceback.print_exc(file=sys.stdout)
        return 1
    else:
        print('change sent successfully')
        return 0