from msrest.serialization import Model

class FeedbackRecordsDTO(Model):
    """Active learning feedback records.

    :param feedback_records: List of feedback records.
    :type feedback_records:
     list[~azure.cognitiveservices.knowledge.qnamaker.models.FeedbackRecordDTO]
    """
    _attribute_map = {'feedback_records': {'key': 'feedbackRecords', 'type': '[FeedbackRecordDTO]'}}

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(FeedbackRecordsDTO, self).__init__(**kwargs)
        self.feedback_records = kwargs.get('feedback_records', None)