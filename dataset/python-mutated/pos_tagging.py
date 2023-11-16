from nlp_architect.models import chunker
from bigdl.orca.tfpark.text.keras.text_model import TextKerasModel
from bigdl.dllib.utils.log4Error import *

class SequenceTagger(TextKerasModel):
    """
    The model used as POS-tagger and chunker for sentence tagging, which contains three
    Bidirectional LSTM layers.

    This model can have one or two input(s):
    - word indices of shape (batch, sequence_length)
    *If char_vocab_size is not None:
    - character indices of shape (batch, sequence_length, word_length)
    This model has two outputs:
    - pos tags of shape (batch, sequence_length, num_pos_labels)
    - chunk tags of shape (batch, sequence_length, num_chunk_labels)

    :param num_pos_labels: Positive int. The number of pos labels to be classified.
    :param num_chunk_labels: Positive int. The number of chunk labels to be classified.
    :param word_vocab_size: Positive int. The size of the word dictionary.
    :param char_vocab_size: Positive int. The size of the character dictionary.
                            Default is None and in this case only one input, namely word indices
                            is expected.
    :param word_length: Positive int. The max word length in characters. Default is 12.
    :param feature_size: Positive int. The size of Embedding and Bi-LSTM layers. Default is 100.
    :param dropout: Dropout rate. Default is 0.5.
    :param classifier: String. The classification layer used for tagging chunks.
                       Either 'softmax' or 'crf' (Conditional Random Field). Default is 'softmax'.
    :param optimizer: Optimizer to train the model. If not specified, it will by default
                      to be tf.train.AdamOptimizer().
    """

    def __init__(self, num_pos_labels, num_chunk_labels, word_vocab_size, char_vocab_size=None, word_length=12, feature_size=100, dropout=0.2, classifier='softmax', optimizer=None):
        if False:
            while True:
                i = 10
        classifier = classifier.lower()
        invalidInputError(classifier in ['softmax', 'crf'], 'classifier should be either softmax or crf')
        super(SequenceTagger, self).__init__(chunker.SequenceTagger(use_cudnn=False), vocabulary_size=word_vocab_size, num_pos_labels=num_pos_labels, num_chunk_labels=num_chunk_labels, char_vocab_size=char_vocab_size, max_word_len=word_length, feature_size=feature_size, dropout=dropout, classifier=classifier, optimizer=optimizer)

    @staticmethod
    def load_model(path):
        if False:
            i = 10
            return i + 15
        '\n        Load an existing SequenceTagger model (with weights) from HDF5 file.\n\n        :param path: String. The path to the pre-defined model.\n        :return: NER.\n        '
        labor = chunker.SequenceTagger(use_cudnn=False)
        model = TextKerasModel._load_model(labor, path)
        model.__class__ = SequenceTagger
        return model