from __future__ import annotations
DOCUMENTATION = '\n  name: urlsplit\n  version_added: "2.4"\n  short_description: get components from URL\n  description:\n    - Split a URL into its component parts.\n  positional: _input, query\n  options:\n    _input:\n      description: URL string to split.\n      type: str\n      required: true\n    query:\n      description: Specify a single component to return.\n      type: str\n      choices: ["fragment", "hostname", "netloc", "password",  "path",  "port",  "query", "scheme",  "username"]\n'
EXAMPLES = '\n\n    parts: \'{{ "http://user:password@www.acme.com:9000/dir/index.html?query=term#fragment" | urlsplit }}\'\n    # =>\n    #   {\n    #       "fragment": "fragment",\n    #       "hostname": "www.acme.com",\n    #       "netloc": "user:password@www.acme.com:9000",\n    #       "password": "password",\n    #       "path": "/dir/index.html",\n    #       "port": 9000,\n    #       "query": "query=term",\n    #       "scheme": "http",\n    #       "username": "user"\n    #   }\n\n    hostname: \'{{ "http://user:password@www.acme.com:9000/dir/index.html?query=term#fragment" | urlsplit("hostname") }}\'\n    # => \'www.acme.com\'\n\n    query: \'{{ "http://user:password@www.acme.com:9000/dir/index.html?query=term#fragment" | urlsplit("query") }}\'\n    # => \'query=term\'\n\n    path: \'{{ "http://user:password@www.acme.com:9000/dir/index.html?query=term#fragment" | urlsplit("path") }}\'\n    # => \'/dir/index.html\'\n'
RETURN = '\n  _value:\n    description:\n      - A dictionary with components as keyword and their value.\n      - If O(query) is provided, a string or integer will be returned instead, depending on O(query).\n    type: any\n'
from urllib.parse import urlsplit
from ansible.errors import AnsibleFilterError
from ansible.utils import helpers

def split_url(value, query='', alias='urlsplit'):
    if False:
        i = 10
        return i + 15
    results = helpers.object_to_dict(urlsplit(value), exclude=['count', 'index', 'geturl', 'encode'])
    if query:
        if query not in results:
            raise AnsibleFilterError(alias + ': unknown URL component: %s' % query)
        return results[query]
    else:
        return results

class FilterModule(object):
    """ URI filter """

    def filters(self):
        if False:
            return 10
        return {'urlsplit': split_url}