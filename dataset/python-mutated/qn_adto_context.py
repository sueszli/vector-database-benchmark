from .context_dto import ContextDTO

class QnADTOContext(ContextDTO):
    """Context of a QnA.

    :param is_context_only: To mark if a prompt is relevant only with a
     previous question or not.
     true - Do not include this QnA as search result for queries without
     context
     false - ignores context and includes this QnA in search result
    :type is_context_only: bool
    :param prompts: List of prompts associated with the answer.
    :type prompts:
     list[~azure.cognitiveservices.knowledge.qnamaker.models.PromptDTO]
    """
    _validation = {'prompts': {'max_items': 20}}
    _attribute_map = {'is_context_only': {'key': 'isContextOnly', 'type': 'bool'}, 'prompts': {'key': 'prompts', 'type': '[PromptDTO]'}}

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(QnADTOContext, self).__init__(**kwargs)