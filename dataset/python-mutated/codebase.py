from twisted.internet import defer

class AbsoluteSourceStampsMixin:
    _lastCodebases = None

    @defer.inlineCallbacks
    def getCodebaseDict(self, codebase):
        if False:
            return 10
        assert self.codebases
        if self._lastCodebases is None:
            self._lastCodebases = (yield self.getState('lastCodebases', {}))
        return self._lastCodebases.get(codebase, self.codebases[codebase])

    @defer.inlineCallbacks
    def recordChange(self, change):
        if False:
            return 10
        codebase = (yield self.getCodebaseDict(change.codebase))
        lastChange = codebase.get('lastChange', -1)
        if change.number > lastChange:
            self._lastCodebases[change.codebase] = {'repository': change.repository, 'branch': change.branch, 'revision': change.revision, 'lastChange': change.number}
            yield self.setState('lastCodebases', self._lastCodebases)