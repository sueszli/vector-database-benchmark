import inspect
from twisted.internet import defer
from buildbot.data.exceptions import InvalidPathError
from buildbot.util import bytes2unicode

class EndpointMatcherBase:

    def __init__(self, role, defaultDeny=True):
        if False:
            for i in range(10):
                print('nop')
        self.role = role
        self.defaultDeny = defaultDeny
        self.owner = None

    def setAuthz(self, authz):
        if False:
            i = 10
            return i + 15
        self.authz = authz
        self.master = authz.master

    def match(self, ep, action='get', options=None):
        if False:
            return 10
        if options is None:
            options = {}
        try:
            (epobject, epdict) = self.master.data.getEndpoint(ep)
            for klass in inspect.getmro(epobject.__class__):
                m = getattr(self, 'match_' + klass.__name__ + '_' + action, None)
                if m is not None:
                    return m(epobject, epdict, options)
                m = getattr(self, 'match_' + klass.__name__, None)
                if m is not None:
                    return m(epobject, epdict, options)
        except InvalidPathError:
            return defer.succeed(None)
        return defer.succeed(None)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        args = []
        for (k, v) in self.__dict__.items():
            if isinstance(v, str):
                args.append(f"{k}='{v}'")
        return f"{self.__class__.__name__}({', '.join(args)})"

class Match:

    def __init__(self, master, build=None, buildrequest=None, buildset=None):
        if False:
            while True:
                i = 10
        self.master = master
        self.build = build
        self.buildrequest = buildrequest
        self.buildset = buildset

    def getOwner(self):
        if False:
            print('Hello World!')
        if self.buildset:
            return self.getOwnerFromBuildset(self.buildset)
        elif self.buildrequest:
            return self.getOwnerFromBuildRequest(self.buildrequest)
        elif self.build:
            return self.getOwnerFromBuild(self.build)
        return defer.succeed(None)

    @defer.inlineCallbacks
    def getOwnerFromBuild(self, build):
        if False:
            return 10
        br = (yield self.master.data.get(('buildrequests', build['buildrequestid'])))
        owner = (yield self.getOwnerFromBuildRequest(br))
        return owner

    @defer.inlineCallbacks
    def getOwnerFromBuildsetOrBuildRequest(self, buildsetorbuildrequest):
        if False:
            while True:
                i = 10
        props = (yield self.master.data.get(('buildsets', buildsetorbuildrequest['buildsetid'], 'properties')))
        if 'owner' in props:
            return props['owner'][0]
        return None
    getOwnerFromBuildRequest = getOwnerFromBuildsetOrBuildRequest
    getOwnerFromBuildSet = getOwnerFromBuildsetOrBuildRequest

class AnyEndpointMatcher(EndpointMatcherBase):

    def match(self, ep, action='get', options=None):
        if False:
            print('Hello World!')
        return defer.succeed(Match(self.master))

class AnyControlEndpointMatcher(EndpointMatcherBase):

    def match(self, ep, action='', options=None):
        if False:
            while True:
                i = 10
        if bytes2unicode(action).lower() != 'get':
            return defer.succeed(Match(self.master))
        return defer.succeed(None)

class StopBuildEndpointMatcher(EndpointMatcherBase):

    def __init__(self, builder=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.builder = builder
        super().__init__(**kwargs)

    @defer.inlineCallbacks
    def matchFromBuilderId(self, builderid):
        if False:
            return 10
        builder = (yield self.master.data.get(('builders', builderid)))
        buildername = builder['name']
        return self.authz.match(buildername, self.builder)

    @defer.inlineCallbacks
    def match_BuildEndpoint_stop(self, epobject, epdict, options):
        if False:
            print('Hello World!')
        build = (yield epobject.get({}, epdict))
        if self.builder is None:
            return Match(self.master, build=build)
        if build is not None:
            ret = (yield self.matchFromBuilderId(build['builderid']))
            if ret:
                return Match(self.master, build=build)
        return None

    @defer.inlineCallbacks
    def match_BuildRequestEndpoint_stop(self, epobject, epdict, options):
        if False:
            while True:
                i = 10
        buildrequest = (yield epobject.get({}, epdict))
        if self.builder is None:
            return Match(self.master, buildrequest=buildrequest)
        if buildrequest is not None:
            ret = (yield self.matchFromBuilderId(buildrequest['builderid']))
            if ret:
                return Match(self.master, buildrequest=buildrequest)
        return None

class ForceBuildEndpointMatcher(EndpointMatcherBase):

    def __init__(self, builder=None, **kwargs):
        if False:
            return 10
        self.builder = builder
        super().__init__(**kwargs)

    @defer.inlineCallbacks
    def match_ForceSchedulerEndpoint_force(self, epobject, epdict, options):
        if False:
            while True:
                i = 10
        if self.builder is None:
            return Match(self.master)
        sched = (yield epobject.findForceScheduler(epdict['schedulername']))
        if sched is not None:
            builderNames = options.get('builderNames')
            builderid = options.get('builderid')
            builderNames = (yield sched.computeBuilderNames(builderNames, builderid))
            for buildername in builderNames:
                if self.authz.match(buildername, self.builder):
                    return Match(self.master)
        return None

class RebuildBuildEndpointMatcher(EndpointMatcherBase):

    def __init__(self, builder=None, **kwargs):
        if False:
            print('Hello World!')
        self.builder = builder
        super().__init__(**kwargs)

    @defer.inlineCallbacks
    def matchFromBuilderId(self, builderid):
        if False:
            i = 10
            return i + 15
        builder = (yield self.master.data.get(('builders', builderid)))
        buildername = builder['name']
        return self.authz.match(buildername, self.builder)

    @defer.inlineCallbacks
    def match_BuildEndpoint_rebuild(self, epobject, epdict, options):
        if False:
            for i in range(10):
                print('nop')
        build = (yield epobject.get({}, epdict))
        if self.builder is None:
            return Match(self.master, build=build)
        if build is not None:
            ret = (yield self.matchFromBuilderId(build['builderid']))
            if ret:
                return Match(self.master, build=build)
        return None

class EnableSchedulerEndpointMatcher(EndpointMatcherBase):

    def match_SchedulerEndpoint_enable(self, epobject, epdict, options):
        if False:
            for i in range(10):
                print('nop')
        return defer.succeed(Match(self.master))

class ViewBuildsEndpointMatcher(EndpointMatcherBase):

    def __init__(self, branch=None, project=None, builder=None, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.branch = branch
        self.project = project
        self.builder = builder

class BranchEndpointMatcher(EndpointMatcherBase):

    def __init__(self, branch, **kwargs):
        if False:
            while True:
                i = 10
        self.branch = branch
        super().__init__(**kwargs)