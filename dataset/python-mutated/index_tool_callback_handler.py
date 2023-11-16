from typing import List
from langchain.schema import Document
from core.conversation_message_task import ConversationMessageTask
from extensions.ext_database import db
from models.dataset import DocumentSegment

class DatasetIndexToolCallbackHandler:
    """Callback handler for dataset tool."""

    def __init__(self, dataset_id: str, conversation_message_task: ConversationMessageTask) -> None:
        if False:
            while True:
                i = 10
        self.dataset_id = dataset_id
        self.conversation_message_task = conversation_message_task

    def on_tool_end(self, documents: List[Document]) -> None:
        if False:
            return 10
        'Handle tool end.'
        for document in documents:
            doc_id = document.metadata['doc_id']
            db.session.query(DocumentSegment).filter(DocumentSegment.dataset_id == self.dataset_id, DocumentSegment.index_node_id == doc_id).update({DocumentSegment.hit_count: DocumentSegment.hit_count + 1}, synchronize_session=False)
            db.session.commit()

    def return_retriever_resource_info(self, resource: List):
        if False:
            for i in range(10):
                print('nop')
        'Handle return_retriever_resource_info.'
        self.conversation_message_task.on_dataset_query_finish(resource)