from uuid import uuid4
from shortGPT.database.db_document import TINY_MONGO_DATABASE, TinyMongoDocument
from shortGPT.database.content_data_manager import ContentDataManager

class ContentDatabase:

    def __init__(self):
        if False:
            print('Hello World!')
        self.content_collection = TINY_MONGO_DATABASE['content_db']['content_documents']

    def instanciateContentDataManager(self, id: str, content_type: str, new=False):
        if False:
            for i in range(10):
                print('nop')
        db_doc = TinyMongoDocument('content_db', 'content_documents', id)
        return ContentDataManager(db_doc, content_type, new)

    def getContentDataManager(self, id, content_type: str):
        if False:
            print('Hello World!')
        try:
            db_doc = TinyMongoDocument('content_db', 'content_documents', id)
            return ContentDataManager(db_doc, content_type, False)
        except:
            return None

    def createContentDataManager(self, content_type: str) -> ContentDataManager:
        if False:
            while True:
                i = 10
        try:
            new_short_id = uuid4().hex[:24]
            db_doc = TinyMongoDocument('content_db', 'content_documents', new_short_id, True)
            return ContentDataManager(db_doc, content_type, True)
        except:
            return None