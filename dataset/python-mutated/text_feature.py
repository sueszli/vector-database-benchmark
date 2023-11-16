import sys
import six
from bigdl.dllib.utils.common import JavaValue
from bigdl.dllib.utils.file_utils import callZooFunc
from bigdl.dllib.utils.log4Error import *
if sys.version >= '3':
    long = int
    unicode = str

class TextFeature(JavaValue):
    """
    Each TextFeature keeps information of a single text record.
    It can include various status (if any) of a text,
    e.g. original text content, uri, category label, tokens, index representation
    of tokens, BigDL Sample representation, prediction result and so on.
    """

    def __init__(self, text=None, label=None, uri=None, jvalue=None, bigdl_type='float'):
        if False:
            return 10
        if text is not None:
            invalidInputError(isinstance(text, six.string_types), 'text of a TextFeature should be a string')
        if uri is not None:
            invalidInputError(isinstance(uri, six.string_types), 'uri of a TextFeature should be a string')
        if label is not None:
            super(TextFeature, self).__init__(jvalue, bigdl_type, text, int(label), uri)
        else:
            super(TextFeature, self).__init__(jvalue, bigdl_type, text, uri)

    def get_text(self):
        if False:
            print('Hello World!')
        '\n        Get the text content of the TextFeature.\n\n        :return: String\n        '
        return callZooFunc(self.bigdl_type, 'textFeatureGetText', self.value)

    def get_label(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the label of the TextFeature.\n        If no label is stored, -1 will be returned.\n\n        :return: Int\n        '
        return callZooFunc(self.bigdl_type, 'textFeatureGetLabel', self.value)

    def get_uri(self):
        if False:
            print('Hello World!')
        '\n        Get the identifier of the TextFeature.\n        If no id is stored, None will be returned.\n\n        :return: String\n        '
        return callZooFunc(self.bigdl_type, 'textFeatureGetURI', self.value)

    def has_label(self):
        if False:
            while True:
                i = 10
        '\n        Whether the TextFeature contains label.\n\n        :return: Boolean\n        '
        return callZooFunc(self.bigdl_type, 'textFeatureHasLabel', self.value)

    def set_label(self, label):
        if False:
            while True:
                i = 10
        '\n        Set the label for the TextFeature.\n\n        :param label: Int\n        :return: The TextFeature with label.\n        '
        self.value = callZooFunc(self.bigdl_type, 'textFeatureSetLabel', self.value, int(label))
        return self

    def get_tokens(self):
        if False:
            return 10
        "\n        Get the tokens of the TextFeature.\n        If text hasn't been segmented, None will be returned.\n\n        :return: List of String\n        "
        return callZooFunc(self.bigdl_type, 'textFeatureGetTokens', self.value)

    def get_sample(self):
        if False:
            while True:
                i = 10
        "\n        Get the Sample representation of the TextFeature.\n        If the TextFeature hasn't been transformed to Sample, None will be returned.\n\n        :return: BigDL Sample\n        "
        return callZooFunc(self.bigdl_type, 'textFeatureGetSample', self.value)

    def keys(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the keys that the TextFeature contains.\n\n        :return: List of String\n        '
        return callZooFunc(self.bigdl_type, 'textFeatureGetKeys', self.value)