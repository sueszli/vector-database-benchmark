from abc import abstractmethod, ABCMeta
from future.utils import with_metaclass
from snips_nlu.common.abc_utils import classproperty
from snips_nlu.pipeline.processing_unit import ProcessingUnit

class SlotFiller(with_metaclass(ABCMeta, ProcessingUnit)):
    """Abstraction which performs slot filling

    A custom slot filler must inherit this class to be used in a
    :class:`.ProbabilisticIntentParser`
    """

    @classproperty
    def unit_name(cls):
        if False:
            while True:
                i = 10
        return SlotFiller.registered_name(cls)

    @abstractmethod
    def fit(self, dataset, intent):
        if False:
            i = 10
            return i + 15
        'Fit the slot filler with a valid Snips dataset'
        pass

    @abstractmethod
    def get_slots(self, text):
        if False:
            for i in range(10):
                print('nop')
        'Performs slot extraction (slot filling) on the provided *text*\n\n        Returns:\n            list of dict: The list of extracted slots. See\n            :func:`.unresolved_slot` for the output format of a slot\n        '
        pass