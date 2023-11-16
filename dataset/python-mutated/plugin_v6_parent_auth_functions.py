from typing import Optional
from ckan.types import AuthResult, Context, DataDict
import ckan.plugins as plugins

def package_delete(context: Context, data_dict: Optional[DataDict]=None) -> AuthResult:
    if False:
        while True:
            i = 10
    return {'success': False, 'msg': 'Only sysadmins can delete packages'}

class ExampleIAuthFunctionsPlugin(plugins.SingletonPlugin):
    plugins.implements(plugins.IAuthFunctions)

    def get_auth_functions(self):
        if False:
            print('Hello World!')
        return {'package_delete': package_delete}