from dataclasses import dataclass
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from . import register_task

@dataclass
class TranslationFromPretrainedXLMConfig(TranslationConfig):
    pass

@register_task('translation_from_pretrained_xlm', dataclass=TranslationFromPretrainedXLMConfig)
class TranslationFromPretrainedXLMTask(TranslationTask):
    """
    Same as TranslationTask except use the MaskedLMDictionary class so that
    we can load data that was binarized with the MaskedLMDictionary class.

    This task should be used for the entire training pipeline when we want to
    train an NMT model from a pretrained XLM checkpoint: binarizing NMT data,
    training NMT with the pretrained XLM checkpoint, and subsequent evaluation
    of that trained model.
    """

    @classmethod
    def load_dictionary(cls, filename):
        if False:
            print('Hello World!')
        'Load the masked LM dictionary from the filename\n\n        Args:\n            filename (str): the filename\n        '
        return MaskedLMDictionary.load(filename)