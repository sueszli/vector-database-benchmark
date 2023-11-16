from typing import List, Dict
import numpy
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import FlagField, TextField, SequenceLabelField
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

@Predictor.register('sentence_tagger')
class SentenceTaggerPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single set of tags for it.  In particular, it can be used with
    the [`CrfTagger`](https://docs.allennlp.org/models/main/models/tagging/models/crf_tagger/)
    model and also the [`SimpleTagger`](../models/simple_tagger.md) model.

    Registered as a `Predictor` with name "sentence_tagger".
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str='en_core_web_sm') -> None:
        if False:
            return 10
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyTokenizer(language=language)

    def predict(self, sentence: str) -> JsonDict:
        if False:
            return 10
        return self.predict_json({'sentence': sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        if False:
            return 10
        '\n        Expects JSON that looks like `{"sentence": "..."}`.\n        Runs the underlying model, and adds the `"words"` to the output.\n        '
        sentence = json_dict['sentence']
        tokens = self._tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(tokens)

    def predictions_to_labeled_instances(self, instance: Instance, outputs: Dict[str, numpy.ndarray]) -> List[Instance]:
        if False:
            i = 10
            return i + 15
        '\n        This function currently only handles BIOUL tags.\n\n        Imagine an NER model predicts three named entities (each one with potentially\n        multiple tokens). For each individual entity, we create a new Instance that has\n        the label set to only that entity and the rest of the tokens are labeled as outside.\n        We then return a list of those Instances.\n\n        For example:\n\n        ```text\n        Mary  went to Seattle to visit Microsoft Research\n        U-Per  O    O   U-Loc  O   O     B-Org     L-Org\n        ```\n\n        We create three instances.\n\n        ```text\n        Mary  went to Seattle to visit Microsoft Research\n        U-Per  O    O    O     O   O       O         O\n\n        Mary  went to Seattle to visit Microsoft Research\n        O      O    O   U-LOC  O   O       O         O\n\n        Mary  went to Seattle to visit Microsoft Research\n        O      O    O    O     O   O     B-Org     L-Org\n        ```\n\n        We additionally add a flag to these instances to tell the model to only compute loss on\n        non-O tags, so that we get gradients that are specific to the particular span prediction\n        that each instance represents.\n        '
        predicted_tags = outputs['tags']
        predicted_spans = []
        i = 0
        while i < len(predicted_tags):
            tag = predicted_tags[i]
            if tag[0] == 'U':
                current_tags = [t if idx == i else 'O' for (idx, t) in enumerate(predicted_tags)]
                predicted_spans.append(current_tags)
            elif tag[0] == 'B':
                begin_idx = i
                while tag[0] != 'L':
                    i += 1
                    tag = predicted_tags[i]
                end_idx = i
                current_tags = [t if begin_idx <= idx <= end_idx else 'O' for (idx, t) in enumerate(predicted_tags)]
                predicted_spans.append(current_tags)
            i += 1
        instances = []
        for labels in predicted_spans:
            new_instance = instance.duplicate()
            text_field: TextField = instance['tokens']
            new_instance.add_field('tags', SequenceLabelField(labels, text_field), self._model.vocab)
            new_instance.add_field('ignore_loss_on_o_tags', FlagField(True))
            instances.append(new_instance)
        return instances