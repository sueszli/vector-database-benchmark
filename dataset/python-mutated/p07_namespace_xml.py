"""
Topic: 处理含命名空间的XML文档
Desc : 
"""

class XMLNamespaces:

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.namespaces = {}
        for (name, uri) in kwargs.items():
            self.register(name, uri)

    def register(self, name, uri):
        if False:
            return 10
        self.namespaces[name] = '{' + uri + '}'

    def __call__(self, path):
        if False:
            print('Hello World!')
        return path.format_map(self.namespaces)
if __name__ == '__main__':
    ns = XMLNamespaces(html='http://www.w3.org/1999/xhtml')