from twisted.internet import defer
from buildbot.data import base
from buildbot.data import types
from buildbot.schedulers import forcesched
from buildbot.www.rest import JSONRPC_CODES
from buildbot.www.rest import BadJsonRpc2

def forceScheduler2Data(sched):
    if False:
        i = 10
        return i + 15
    ret = {'all_fields': [], 'name': str(sched.name), 'button_name': str(sched.buttonName), 'label': str(sched.label), 'builder_names': [str(name) for name in sched.builderNames], 'enabled': sched.enabled}
    ret['all_fields'] = [field.getSpec() for field in sched.all_fields]
    return ret

class ForceSchedulerEndpoint(base.Endpoint):
    kind = base.EndpointKind.SINGLE
    pathPatterns = '\n        /forceschedulers/i:schedulername\n    '

    def findForceScheduler(self, schedulername):
        if False:
            while True:
                i = 10
        for sched in self.master.allSchedulers():
            if sched.name == schedulername and isinstance(sched, forcesched.ForceScheduler):
                return defer.succeed(sched)
        return None

    @defer.inlineCallbacks
    def get(self, resultSpec, kwargs):
        if False:
            i = 10
            return i + 15
        sched = (yield self.findForceScheduler(kwargs['schedulername']))
        if sched is not None:
            return forceScheduler2Data(sched)
        return None

    @defer.inlineCallbacks
    def control(self, action, args, kwargs):
        if False:
            print('Hello World!')
        if action == 'force':
            sched = (yield self.findForceScheduler(kwargs['schedulername']))
            if 'owner' not in args:
                args['owner'] = 'user'
            try:
                res = (yield sched.force(**args))
                return res
            except forcesched.CollectedValidationError as e:
                raise BadJsonRpc2(e.errors, JSONRPC_CODES['invalid_params']) from e
        return None

class ForceSchedulersEndpoint(base.Endpoint):
    kind = base.EndpointKind.COLLECTION
    pathPatterns = '\n        /forceschedulers\n        /builders/:builderid/forceschedulers\n    '
    rootLinkName = 'forceschedulers'

    @defer.inlineCallbacks
    def get(self, resultSpec, kwargs):
        if False:
            while True:
                i = 10
        ret = []
        builderid = kwargs.get('builderid', None)
        if builderid is not None:
            bdict = (yield self.master.db.builders.getBuilder(builderid))
        for sched in self.master.allSchedulers():
            if isinstance(sched, forcesched.ForceScheduler):
                if builderid is not None and bdict['name'] not in sched.builderNames:
                    continue
                ret.append(forceScheduler2Data(sched))
        return ret

class ForceScheduler(base.ResourceType):
    name = 'forcescheduler'
    plural = 'forceschedulers'
    endpoints = [ForceSchedulerEndpoint, ForceSchedulersEndpoint]
    keyField = 'name'

    class EntityType(types.Entity):
        name = types.Identifier(50)
        button_name = types.String()
        label = types.String()
        builder_names = types.List(of=types.Identifier(50))
        enabled = types.Boolean()
        all_fields = types.List(of=types.JsonObject())
    entityType = EntityType(name, 'Forcescheduler')