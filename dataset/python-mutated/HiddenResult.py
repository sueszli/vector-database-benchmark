from coalib.results.Result import Result
from coalib.results.RESULT_SEVERITY import RESULT_SEVERITY
from coalib.bearlib.aspects import aspectbase
from coala_utils.decorators import enforce_signature, generate_ordering, generate_repr

@generate_repr(('id', hex), 'origin', 'message', 'contents')
@generate_ordering('contents', 'origin', 'message_base')
class HiddenResult(Result):
    """
    This is a result that is not meant to be shown to the user. It can be used
    to transfer any data from a dependent bear to others.
    """

    @enforce_signature
    def __init__(self, origin, contents, message: str='', affected_code: (tuple, list)=(), severity: int=RESULT_SEVERITY.NORMAL, additional_info: str='', debug_msg='', diffs: (dict, None)=None, confidence: int=100, aspect: (aspectbase, None)=None, message_arguments: dict={}, applied_actions: dict={}):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a new HiddenResult. The contents can be accessed with\n        obj.contents later.\n\n        :param origin:\n            The originating bear.\n        :param contents:\n            Any object to send additional data (arbitrary python objects)\n            besides a message to the dependent bear. The data has to\n            be picklable.\n        :param affected_code:\n            A tuple of ``SourceRange`` objects pointing to related positions in\n            the source code.\n        :param severity:\n            Severity of this result.\n        :param additional_info:\n            A long description holding additional information about the issue\n            and/or how to fix it. You can use this like a manual entry for a\n            category of issues.\n        :param debug_msg:\n            A message which may help the user find out why this result was\n            yielded.\n        :param diffs:\n            A dictionary with filename as key and ``Diff`` object\n            associated with it as value.\n        :param confidence:\n            A number between 0 and 100 describing the likelihood of this result\n            being a real issue.\n        :param aspect:\n            An aspectclass instance which this result is associated to.\n            Note that this should be a leaf of the aspect tree!\n            (If you have a node, spend some time figuring out which of\n            the leafs exactly your result belongs to.)\n        :param message_arguments:\n            Arguments to be provided to the base message.\n        :param applied_actions:\n            A dictionary that contains the result, file_dict, file_diff_dict and\n            the section for an action.\n        '
        Result.__init__(self, origin, message, affected_code, severity, additional_info, debug_msg, diffs, confidence, aspect, message_arguments, applied_actions)
        self.contents = contents