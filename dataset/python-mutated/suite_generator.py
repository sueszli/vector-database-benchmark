from datetime import datetime

class EmbeddedDoc:

    def __init__(self, _id):
        if False:
            for i in range(10):
                print('nop')
        self._id = _id
        self._created = datetime.utcnow()