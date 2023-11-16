from __future__ import absolute_import
'Server-side pack repository related request implmentations.'
from bzrlib.smart.request import FailedSmartServerResponse, SuccessfulSmartServerResponse
from bzrlib.smart.repository import SmartServerRepositoryRequest

class SmartServerPackRepositoryAutopack(SmartServerRepositoryRequest):

    def do_repository_request(self, repository):
        if False:
            i = 10
            return i + 15
        pack_collection = getattr(repository, '_pack_collection', None)
        if pack_collection is None:
            return SuccessfulSmartServerResponse(('ok',))
        repository.lock_write()
        try:
            repository._pack_collection.autopack()
        finally:
            repository.unlock()
        return SuccessfulSmartServerResponse(('ok',))