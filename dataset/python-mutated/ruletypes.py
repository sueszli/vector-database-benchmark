from mongoengine import ValidationError
import six
from st2common import log as logging
from st2common.models.api.rule import RuleTypeAPI
from st2common.persistence.rule import RuleType
from st2common.router import abort
http_client = six.moves.http_client
LOG = logging.getLogger(__name__)

class RuleTypesController(object):
    """
    Implements the RESTful web endpoint that handles
    the lifecycle of a RuleType in the system.
    """

    @staticmethod
    def __get_by_id(id):
        if False:
            while True:
                i = 10
        try:
            return RuleType.get_by_id(id)
        except (ValueError, ValidationError) as e:
            msg = 'Database lookup for id="%s" resulted in exception. %s' % (id, e)
            LOG.exception(msg)
            abort(http_client.NOT_FOUND, msg)

    @staticmethod
    def __get_by_name(name):
        if False:
            return 10
        try:
            return [RuleType.get_by_name(name)]
        except ValueError as e:
            LOG.debug('Database lookup for name="%s" resulted in exception : %s.', name, e)
            return []

    def get_one(self, id):
        if False:
            while True:
                i = 10
        '\n        List RuleType objects by id.\n\n        Handle:\n            GET /ruletypes/1\n        '
        ruletype_db = RuleTypesController.__get_by_id(id)
        ruletype_api = RuleTypeAPI.from_model(ruletype_db)
        return ruletype_api

    def get_all(self):
        if False:
            while True:
                i = 10
        '\n        List all RuleType objects.\n\n        Handles requests:\n            GET /ruletypes/\n        '
        ruletype_dbs = RuleType.get_all()
        ruletype_apis = [RuleTypeAPI.from_model(runnertype_db) for runnertype_db in ruletype_dbs]
        return ruletype_apis
rule_types_controller = RuleTypesController()