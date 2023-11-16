import six
from st2api.controllers.resource import ResourceController
from st2api.controllers.v1.packs import packs_controller
from st2common.services import packs as packs_service
from st2common.models.api.pack import ConfigSchemaAPI
from st2common.persistence.pack import ConfigSchema
http_client = six.moves.http_client
__all__ = ['PackConfigSchemasController']

class PackConfigSchemasController(ResourceController):
    model = ConfigSchemaAPI
    access = ConfigSchema
    supported_filters = {}

    def __init__(self):
        if False:
            print('Hello World!')
        super(PackConfigSchemasController, self).__init__()
        self.get_one_db_method = packs_service.get_pack_by_ref

    def get_all(self, sort=None, offset=0, limit=None, requester_user=None, **raw_filters):
        if False:
            while True:
                i = 10
        '\n        Retrieve config schema for all the packs.\n\n        Handles requests:\n            GET /config_schema/\n        '
        return super(PackConfigSchemasController, self)._get_all(sort=sort, offset=offset, limit=limit, raw_filters=raw_filters, requester_user=requester_user)

    def get_one(self, pack_ref, requester_user):
        if False:
            while True:
                i = 10
        '\n        Retrieve config schema for a particular pack.\n\n        Handles requests:\n            GET /config_schema/<pack_ref>\n        '
        packs_controller._get_one_by_ref_or_id(ref_or_id=pack_ref, requester_user=requester_user)
        return self._get_one_by_pack_ref(pack_ref=pack_ref)
pack_config_schema_controller = PackConfigSchemasController()