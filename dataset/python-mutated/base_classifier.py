from abc import ABC, abstractmethod
import logging
import torch
import torch.nn as nn
from stanza.models.common.utils import split_into_batches, sort_with_indices, unsort
'\nA base classifier type\n\nCurrently, has the ability to process text or other inputs in a manner\nsuitable for the particular model type.\nIn other words, the CNNClassifier processes lists of words,\nand the ConstituencyClassifier processes trees\n'
logger = logging.getLogger('stanza')

class BaseClassifier(ABC, nn.Module):

    @abstractmethod
    def extract_sentences(self, doc):
        if False:
            while True:
                i = 10
        '\n        Extract the sentences or the relevant information in the sentences from a document\n        '

    def preprocess_sentences(self, sentences):
        if False:
            i = 10
            return i + 15
        "\n        By default, don't do anything\n        "
        return sentences

    def label_sentences(self, sentences, batch_size=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Given a list of sentences, return the model's results on that text.\n        "
        self.eval()
        sentences = self.preprocess_sentences(sentences)
        if batch_size is None:
            intervals = [(0, len(sentences))]
            orig_idx = None
        else:
            (sentences, orig_idx) = sort_with_indices(sentences, key=len, reverse=True)
            intervals = split_into_batches(sentences, batch_size)
        labels = []
        for interval in intervals:
            if interval[1] - interval[0] == 0:
                continue
            output = self(sentences[interval[0]:interval[1]])
            predicted = torch.argmax(output, dim=1)
            labels.extend(predicted.tolist())
        if orig_idx:
            sentences = unsort(sentences, orig_idx)
            labels = unsort(labels, orig_idx)
        logger.debug('Found labels')
        for (label, sentence) in zip(labels, sentences):
            logger.debug((label, sentence))
        return labels