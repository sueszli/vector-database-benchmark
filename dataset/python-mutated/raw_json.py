"""Implementation of RawJSON
"""

class RawJSON(object):
    """Wrap an encoded JSON document for direct embedding in the output

    """

    def __init__(self, encoded_json):
        if False:
            print('Hello World!')
        self.encoded_json = encoded_json