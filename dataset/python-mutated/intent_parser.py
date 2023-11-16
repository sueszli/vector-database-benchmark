from abc import abstractmethod, ABCMeta
from future.utils import with_metaclass
from snips_nlu.common.abc_utils import classproperty
from snips_nlu.pipeline.processing_unit import ProcessingUnit

class IntentParser(with_metaclass(ABCMeta, ProcessingUnit)):
    """Abstraction which performs intent parsing

    A custom intent parser must inherit this class to be used in a
    :class:`.SnipsNLUEngine`
    """

    @classproperty
    def unit_name(cls):
        if False:
            i = 10
            return i + 15
        return IntentParser.registered_name(cls)

    @abstractmethod
    def fit(self, dataset, force_retrain):
        if False:
            print('Hello World!')
        'Fit the intent parser with a valid Snips dataset\n\n        Args:\n            dataset (dict): valid Snips NLU dataset\n            force_retrain (bool): specify whether or not sub units of the\n            intent parser that may be already trained should be retrained\n        '
        pass

    @abstractmethod
    def parse(self, text, intents, top_n):
        if False:
            i = 10
            return i + 15
        'Performs intent parsing on the provided *text*\n\n        Args:\n            text (str): input\n            intents (str or list of str): if provided, reduces the scope of\n                intent parsing to the provided list of intents\n            top_n (int, optional): when provided, this method will return a\n                list of at most top_n most likely intents, instead of a single\n                parsing result.\n                Note that the returned list can contain less than ``top_n``\n                elements, for instance when the parameter ``intents`` is not\n                None, or when ``top_n`` is greater than the total number of\n                intents.\n\n        Returns:\n            dict or list: the most likely intent(s) along with the extracted\n            slots. See :func:`.parsing_result` and :func:`.extraction_result`\n            for the output format.\n        '
        pass

    @abstractmethod
    def get_intents(self, text):
        if False:
            while True:
                i = 10
        'Performs intent classification on the provided *text* and returns\n        the list of intents ordered by decreasing probability\n\n        The length of the returned list is exactly the number of intents in the\n        dataset + 1 for the None intent\n\n        .. note::\n\n            The probabilities returned along with each intent are not\n            guaranteed to sum to 1.0. They should be considered as scores\n            between 0 and 1.\n        '
        pass

    @abstractmethod
    def get_slots(self, text, intent):
        if False:
            for i in range(10):
                print('nop')
        'Extract slots from a text input, with the knowledge of the intent\n\n        Args:\n            text (str): input\n            intent (str): the intent which the input corresponds to\n\n        Returns:\n            list: the list of extracted slots\n\n        Raises:\n            IntentNotFoundError: when the intent was not part of the training\n                data\n        '
        pass