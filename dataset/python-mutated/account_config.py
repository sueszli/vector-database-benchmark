"""
.. module: security_monkey.views.account_config
    :platform: Unix
    :synopsis: Manages generic AWS account.


.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>


"""
from security_monkey.views import AuthenticatedService
from security_monkey.datastore import AccountType
from security_monkey.account_manager import account_registry, load_all_account_types
from security_monkey import rbac
from flask_restful import reqparse

class AccountConfigGet(AuthenticatedService):
    decorators = [rbac.allow(['View'], ['GET'])]

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.reqparse = reqparse.RequestParser()
        super(AccountConfigGet, self).__init__()

    def get(self, account_fields):
        if False:
            for i in range(10):
                print('nop')
        '\n            .. http:get:: /api/1/account_config/account_fields (all or custom)\n\n            Get a list of Account types\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                GET /api/1/account_config/all HTTP/1.1\n                Host: example.com\n                Accept: application/json, text/javascript\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 200 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    auth: {\n                        authenticated: true,\n                        user: "user@example.com"\n                    }\n                }\n\n            :statuscode 200: no error\n            :statuscode 401: Authentication failure. Please login.\n        '
        load_all_account_types()
        marshaled = {}
        account_types = AccountType.query.all()
        configs_marshaled = {}
        for account_type in account_types:
            acc_manager = account_registry.get(account_type.name)
            if acc_manager is not None:
                values = {}
                values['identifier_label'] = acc_manager.identifier_label
                values['identifier_tool_tip'] = acc_manager.identifier_tool_tip
                fields = []
                if account_fields == 'all':
                    fields.append({'name': 'identifier', 'label': '', 'editable': True, 'tool_tip': '', 'password': False, 'allowed_values': None})
                    fields.append({'name': 'name', 'label': '', 'editable': True, 'tool_tip': '', 'password': False, 'allowed_values': None})
                    fields.append({'name': 'notes', 'label': '', 'editable': True, 'tool_tip': '', 'password': False, 'allowed_values': None})
                for config in acc_manager.custom_field_configs:
                    if account_fields == 'custom' or not config.password:
                        field_marshaled = {'name': config.name, 'label': config.label, 'editable': config.db_item, 'tool_tip': config.tool_tip, 'password': config.password, 'allowed_values': config.allowed_values}
                        fields.append(field_marshaled)
                    values['fields'] = fields
                configs_marshaled[account_type.name] = values
        marshaled['custom_configs'] = configs_marshaled
        marshaled['auth'] = self.auth_dict
        return (marshaled, 200)