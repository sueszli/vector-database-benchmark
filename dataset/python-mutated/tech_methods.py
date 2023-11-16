"""
.. module: security_monkey.views.tech_methods
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>


"""
from security_monkey.views import AuthenticatedService
from security_monkey.auditor import auditor_registry
from security_monkey import rbac

class TechMethodsGet(AuthenticatedService):
    decorators = [rbac.allow(['View'], ['GET'])]

    def __init__(self):
        if False:
            while True:
                i = 10
        super(TechMethodsGet, self).__init__()

    def get(self, tech_ids):
        if False:
            for i in range(10):
                print('nop')
        '\n            .. http:get:: /api/1/techmethods\n\n            Get a list of technologies and associated auditor check methods\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                GET /api/1/techmethods HTTP/1.1\n                Host: example.com\n                Accept: application/json, text/javascript\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 200 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    "technologies": [ "subnet" ]\n                    "tech_methods": { "subnet": [ "check_internet_access" ] }\n                    auth: {\n                        authenticated: true,\n                        user: "user@example.com"\n                    }\n                }\n\n            :statuscode 200: no error\n            :statuscode 401: Authentication failure. Please login.\n        '
        tech_methods = {}
        for key in list(auditor_registry.keys()):
            methods = []
            for auditor_class in auditor_registry[key]:
                auditor = auditor_class('')
                for method_name in dir(auditor):
                    method_name = method_name + ' (' + auditor.__class__.__name__ + ')'
                    if method_name.find('check_') == 0:
                        methods.append(method_name)
                tech_methods[key] = methods
        marshaled_dict = {'tech_methods': tech_methods, 'auth': self.auth_dict}
        return (marshaled_dict, 200)