import re
from datetime import datetime
from twisted.internet import defer
from twisted.python import log
from twisted.web import server
from buildbot.plugins.db import get_plugins
from buildbot.util import bytes2unicode
from buildbot.util import datetime2epoch
from buildbot.util import unicode2bytes
from buildbot.www import resource

class ChangeHookResource(resource.Resource):
    contentType = 'text/html; charset=utf-8'
    children = {}
    needsReconfig = True

    def __init__(self, dialects=None, master=None):
        if False:
            while True:
                i = 10
        "\n        The keys of 'dialects' select a modules to load under\n        master/buildbot/www/hooks/\n        The value is passed to the module's getChanges function, providing\n        configuration options to the dialect.\n        "
        super().__init__(master)
        if dialects is None:
            dialects = {}
        self.dialects = dialects
        self._dialect_handlers = {}
        self.request_dialect = None
        self._plugins = get_plugins('webhooks')

    def reconfigResource(self, new_config):
        if False:
            return 10
        self.dialects = new_config.www.get('change_hook_dialects', {})

    def getChild(self, name, request):
        if False:
            for i in range(10):
                print('nop')
        return self

    def render_GET(self, request):
        if False:
            return 10
        '\n        Responds to events and starts the build process\n          different implementations can decide on what methods they will accept\n        '
        return self.render_POST(request)

    def render_POST(self, request):
        if False:
            while True:
                i = 10
        '\n        Responds to events and starts the build process\n          different implementations can decide on what methods they will accept\n\n        :arguments:\n            request\n                the http request object\n        '
        try:
            d = self.getAndSubmitChanges(request)
        except Exception:
            d = defer.fail()

        def ok(_):
            if False:
                i = 10
                return i + 15
            request.setResponseCode(202)
            request.finish()

        def err(why):
            if False:
                i = 10
                return i + 15
            code = 500
            if why.check(ValueError):
                code = 400
                msg = unicode2bytes(why.getErrorMessage())
            else:
                log.err(why, 'adding changes from web hook')
                msg = b'Error processing changes.'
            request.setResponseCode(code, msg)
            request.write(msg)
            request.finish()
        d.addCallbacks(ok, err)
        return server.NOT_DONE_YET

    @defer.inlineCallbacks
    def getAndSubmitChanges(self, request):
        if False:
            i = 10
            return i + 15
        (changes, src) = (yield self.getChanges(request))
        if not changes:
            request.write(b'no change found')
        else:
            yield self.submitChanges(changes, request, src)
            request.write(unicode2bytes(f'{len(changes)} change found'))

    def makeHandler(self, dialect):
        if False:
            for i in range(10):
                print('nop')
        'create and cache the handler object for this dialect'
        if dialect not in self.dialects:
            m = f"The dialect specified, '{dialect}', wasn't whitelisted in change_hook"
            log.msg(m)
            log.msg("Note: if dialect is 'base' then it's possible your URL is malformed and we didn't regex it properly")
            raise ValueError(m)
        if dialect not in self._dialect_handlers:
            options = self.dialects[dialect]
            if isinstance(options, dict) and 'custom_class' in options:
                klass = options['custom_class']
            else:
                if dialect not in self._plugins:
                    m = f"The dialect specified, '{dialect}', is not registered as a buildbot.webhook plugin"
                    log.msg(m)
                    raise ValueError(m)
                klass = self._plugins.get(dialect)
            self._dialect_handlers[dialect] = klass(self.master, self.dialects[dialect])
        return self._dialect_handlers[dialect]

    @defer.inlineCallbacks
    def getChanges(self, request):
        if False:
            return 10
        '\n        Take the logic from the change hook, and then delegate it\n        to the proper handler\n\n        We use the buildbot plugin mechanisms to find out about dialects\n\n        and call getChanges()\n\n        the return value is a list of changes\n\n        if DIALECT is unspecified, a sample implementation is provided\n        '
        uriRE = re.search('^/change_hook/?([a-zA-Z0-9_]*)', bytes2unicode(request.uri))
        if not uriRE:
            msg = f"URI doesn't match change_hook regex: {request.uri}"
            log.msg(msg)
            raise ValueError(msg)
        changes = []
        src = None
        if uriRE.group(1):
            dialect = uriRE.group(1)
        else:
            dialect = 'base'
        handler = self.makeHandler(dialect)
        (changes, src) = (yield handler.getChanges(request))
        return (changes, src)

    @defer.inlineCallbacks
    def submitChanges(self, changes, request, src):
        if False:
            print('Hello World!')
        for chdict in changes:
            when_timestamp = chdict.get('when_timestamp')
            if isinstance(when_timestamp, datetime):
                chdict['when_timestamp'] = datetime2epoch(when_timestamp)
            for k in ('comments', 'author', 'committer', 'revision', 'branch', 'category', 'revlink', 'repository', 'codebase', 'project'):
                if k in chdict:
                    chdict[k] = bytes2unicode(chdict[k])
            if chdict.get('files'):
                chdict['files'] = [bytes2unicode(f) for f in chdict['files']]
            if chdict.get('properties'):
                chdict['properties'] = dict(((bytes2unicode(k), v) for (k, v) in chdict['properties'].items()))
            chid = (yield self.master.data.updates.addChange(src=bytes2unicode(src), **chdict))
            log.msg(f'injected change {chid}')