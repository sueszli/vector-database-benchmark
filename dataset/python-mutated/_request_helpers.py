def header_property(header_name):
    if False:
        print('Hello World!')
    "Create a read-only header property.\n\n    Args:\n        wsgi_name (str): Case-sensitive name of the header as it would\n            appear in the WSGI environ ``dict`` (i.e., 'HTTP_*')\n\n    Returns:\n        A property instance than can be assigned to a class variable.\n\n    "
    header_name = header_name.lower().encode()

    def fget(self):
        if False:
            return 10
        try:
            return self._asgi_headers[header_name].decode('latin1') or None
        except KeyError:
            return None
    return property(fget)