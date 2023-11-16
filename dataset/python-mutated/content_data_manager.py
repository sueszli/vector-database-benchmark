from shortGPT.database.db_document import AbstractDatabaseDocument

class ContentDataManager:

    def __init__(self, db_doc: AbstractDatabaseDocument, content_type: str, new=False):
        if False:
            i = 10
            return i + 15
        self.contentType = content_type
        self.db_doc = db_doc
        if new:
            self.db_doc._save({'content_type': content_type, 'ready_to_upload': False, 'last_completed_step': 0})

    def save(self, key, value):
        if False:
            while True:
                i = 10
        self.db_doc._save({key: value})

    def get(self, key):
        if False:
            while True:
                i = 10
        return self.db_doc._get(key)

    def _getId(self):
        if False:
            return 10
        return self.db_doc._getId()

    def delete(self):
        if False:
            for i in range(10):
                print('nop')
        self.db_doc.delete()

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.db_doc.__str__()