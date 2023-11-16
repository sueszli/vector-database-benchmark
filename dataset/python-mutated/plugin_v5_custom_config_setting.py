from typing import Optional
from ckan.types import AuthResult, Context, DataDict
import ckan.plugins as plugins
import ckan.plugins.toolkit as toolkit
from ckan.config.declaration import Declaration, Key

def group_create(context: Context, data_dict: Optional[DataDict]=None) -> AuthResult:
    if False:
        print('Hello World!')
    users_can_create_groups = toolkit.config.get('ckan.iauthfunctions.users_can_create_groups')
    if users_can_create_groups:
        return {'success': True}
    else:
        return {'success': False, 'msg': 'Only sysadmins can create groups'}

class ExampleIAuthFunctionsPlugin(plugins.SingletonPlugin):
    plugins.implements(plugins.IAuthFunctions)
    plugins.implements(plugins.IConfigDeclaration)

    def get_auth_functions(self):
        if False:
            return 10
        return {'group_create': group_create}

    def declare_config_options(self, declaration: Declaration, key: Key):
        if False:
            for i in range(10):
                print('nop')
        declaration.declare_bool(key.ckan.iauthfunctions.users_can_create_groups)