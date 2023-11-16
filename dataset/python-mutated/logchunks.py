from twisted.internet import defer
from buildbot.data import base
from buildbot.data import types

class LogChunkEndpointBase(base.BuildNestingMixin, base.Endpoint):

    @defer.inlineCallbacks
    def getLogIdAndDbDictFromKwargs(self, kwargs):
        if False:
            return 10
        if 'logid' in kwargs:
            logid = kwargs['logid']
            dbdict = None
        else:
            stepid = (yield self.getStepid(kwargs))
            if stepid is None:
                return (None, None)
            dbdict = (yield self.master.db.logs.getLogBySlug(stepid, kwargs.get('log_slug')))
            if not dbdict:
                return (None, None)
            logid = dbdict['id']
        return (logid, dbdict)

class LogChunkEndpoint(LogChunkEndpointBase):
    kind = base.EndpointKind.SINGLE
    isPseudoCollection = True
    pathPatterns = '\n        /logchunks\n        /logs/n:logid/contents\n        /steps/n:stepid/logs/i:log_slug/contents\n        /builds/n:buildid/steps/i:step_name/logs/i:log_slug/contents\n        /builds/n:buildid/steps/n:step_number/logs/i:log_slug/contents\n        /builders/n:builderid/builds/n:build_number/steps/i:step_name/logs/i:log_slug/contents\n        /builders/n:builderid/builds/n:build_number/steps/n:step_number/logs/i:log_slug/contents\n    '
    rootLinkName = 'logchunks'

    @defer.inlineCallbacks
    def get(self, resultSpec, kwargs):
        if False:
            return 10
        (logid, dbdict) = (yield self.getLogIdAndDbDictFromKwargs(kwargs))
        if logid is None:
            return None
        firstline = int(resultSpec.offset or 0)
        lastline = None if resultSpec.limit is None else firstline + int(resultSpec.limit) - 1
        resultSpec.removePagination()
        if lastline is None:
            if not dbdict:
                dbdict = (yield self.master.db.logs.getLog(logid))
            if not dbdict:
                return None
            lastline = int(max(0, dbdict['num_lines'] - 1))
        if firstline < 0 or lastline < 0 or firstline > lastline:
            return None
        logLines = (yield self.master.db.logs.getLogLines(logid, firstline, lastline))
        return {'logid': logid, 'firstline': firstline, 'content': logLines}

    def get_kwargs_from_graphql(self, parent, resolve_info, args):
        if False:
            i = 10
            return i + 15
        if parent is not None:
            return self.get_kwargs_from_graphql_parent(parent, resolve_info.parent_type.name)
        return {'logid': args['logid']}

class RawLogChunkEndpoint(LogChunkEndpointBase):
    kind = base.EndpointKind.RAW
    pathPatterns = '\n        /logs/n:logid/raw\n        /steps/n:stepid/logs/i:log_slug/raw\n        /builds/n:buildid/steps/i:step_name/logs/i:log_slug/raw\n        /builds/n:buildid/steps/n:step_number/logs/i:log_slug/raw\n        /builders/n:builderid/builds/n:build_number/steps/i:step_name/logs/i:log_slug/raw\n        /builders/n:builderid/builds/n:build_number/steps/n:step_number/logs/i:log_slug/raw\n    '

    @defer.inlineCallbacks
    def get(self, resultSpec, kwargs):
        if False:
            print('Hello World!')
        (logid, dbdict) = (yield self.getLogIdAndDbDictFromKwargs(kwargs))
        if logid is None:
            return None
        if not dbdict:
            dbdict = (yield self.master.db.logs.getLog(logid))
            if not dbdict:
                return None
        lastline = max(0, dbdict['num_lines'] - 1)
        logLines = (yield self.master.db.logs.getLogLines(logid, 0, lastline))
        if dbdict['type'] == 's':
            logLines = '\n'.join([line[1:] for line in logLines.splitlines()])
        return {'raw': logLines, 'mime-type': 'text/html' if dbdict['type'] == 'h' else 'text/plain', 'filename': dbdict['slug']}

class RawInlineLogChunkEndpoint(LogChunkEndpointBase):
    kind = base.EndpointKind.RAW_INLINE
    pathPatterns = '\n        /logs/n:logid/raw_inline\n        /steps/n:stepid/logs/i:log_slug/raw_inline\n        /builds/n:buildid/steps/i:step_name/logs/i:log_slug/raw_inline\n        /builds/n:buildid/steps/n:step_number/logs/i:log_slug/raw_inline\n        /builders/n:builderid/builds/n:build_number/steps/i:step_name/logs/i:log_slug/raw_inline\n        /builders/n:builderid/builds/n:build_number/steps/n:step_number/logs/i:log_slug/raw_inline\n    '

    @defer.inlineCallbacks
    def get(self, resultSpec, kwargs):
        if False:
            i = 10
            return i + 15
        (logid, dbdict) = (yield self.getLogIdAndDbDictFromKwargs(kwargs))
        if logid is None:
            return None
        if not dbdict:
            dbdict = (yield self.master.db.logs.getLog(logid))
            if not dbdict:
                return None
        lastline = max(0, dbdict['num_lines'] - 1)
        logLines = (yield self.master.db.logs.getLogLines(logid, 0, lastline))
        if dbdict['type'] == 's':
            logLines = '\n'.join([line[1:] for line in logLines.splitlines()])
        return {'raw': logLines, 'mime-type': 'text/html' if dbdict['type'] == 'h' else 'text/plain'}

class LogChunk(base.ResourceType):
    name = 'logchunk'
    plural = 'logchunks'
    endpoints = [LogChunkEndpoint, RawLogChunkEndpoint, RawInlineLogChunkEndpoint]
    keyField = 'logid'

    class EntityType(types.Entity):
        logid = types.Integer()
        firstline = types.Integer()
        content = types.String()
    entityType = EntityType(name, 'LogChunk')