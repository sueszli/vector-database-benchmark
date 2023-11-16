from __future__ import annotations
DOCUMENTATION = "\nname: double_doc\ndescription:\n    - module also uses 'DOCUMENTATION' in class\n"

class Foo:
    DOCUMENTATION = None

    def __init__(self):
        if False:
            return 10
        pass