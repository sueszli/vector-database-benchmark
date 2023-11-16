from abc import abstractmethod, ABCMeta
from future.utils import with_metaclass
from snips_nlu.pipeline.processing_unit import ProcessingUnit
from snips_nlu.common.abc_utils import classproperty

class IntentClassifier(with_metaclass(ABCMeta, ProcessingUnit)):
    """Abstraction which performs intent classification

    A custom intent classifier must inherit this class to be used in a
    :class:`.ProbabilisticIntentParser`
    """

    @classproperty
    def unit_name(cls):
        if False:
            print('Hello World!')
        return IntentClassifier.registered_name(cls)

    @abstractmethod
    def fit(self, dataset):
        if False:
            print('Hello World!')
        'Fit the intent classifier with a valid Snips dataset'
        pass

    @abstractmethod
    def get_intent(self, text, intents_filter):
        if False:
            while True:
                i = 10
        'Performs intent classification on the provided *text*\n\n        Args:\n            text (str): Input\n            intents_filter (str or list of str): When defined, it will find\n                the most likely intent among the list, otherwise it will use\n                the whole list of intents defined in the dataset\n\n        Returns:\n            dict or None: The most likely intent along with its probability or\n            *None* if no intent was found. See\n            :func:`.intent_classification_result` for the output format.\n        '
        pass

    @abstractmethod
    def get_intents(self, text):
        if False:
            i = 10
            return i + 15
        'Performs intent classification on the provided *text* and returns\n        the list of intents ordered by decreasing probability\n\n        The length of the returned list is exactly the number of intents in the\n        dataset + 1 for the None intent\n\n        .. note::\n\n            The probabilities returned along with each intent are not\n            guaranteed to sum to 1.0. They should be considered as scores\n            between 0 and 1.\n        '
        pass