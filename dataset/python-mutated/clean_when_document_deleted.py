from events.document_event import document_was_deleted
from tasks.clean_document_task import clean_document_task

@document_was_deleted.connect
def handle(sender, **kwargs):
    if False:
        print('Hello World!')
    document_id = sender
    dataset_id = kwargs.get('dataset_id')
    clean_document_task.delay(document_id, dataset_id)