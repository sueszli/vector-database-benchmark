import six
from bigdl.dllib.utils.common import JavaValue
from bigdl.dllib.utils.file_utils import callZooFunc
from pyspark import RDD
from bigdl.dllib.utils.log4Error import *

class TextSet(JavaValue):
    """
    TextSet wraps a set of texts with status.
    """

    def __init__(self, jvalue, bigdl_type='float', *args):
        if False:
            for i in range(10):
                print('nop')
        super(TextSet, self).__init__(jvalue, bigdl_type, *args)

    def is_local(self):
        if False:
            i = 10
            return i + 15
        '\n        Whether it is a LocalTextSet.\n\n        :return: Boolean\n        '
        return callZooFunc(self.bigdl_type, 'textSetIsLocal', self.value)

    def is_distributed(self):
        if False:
            return 10
        '\n        Whether it is a DistributedTextSet.\n\n        :return: Boolean\n        '
        return callZooFunc(self.bigdl_type, 'textSetIsDistributed', self.value)

    def to_distributed(self, sc=None, partition_num=4):
        if False:
            print('Hello World!')
        '\n        Convert to a DistributedTextSet.\n\n        Need to specify SparkContext to convert a LocalTextSet to a DistributedTextSet.\n        In this case, you may also want to specify partition_num, the default of which is 4.\n\n        :return: DistributedTextSet\n        '
        if self.is_distributed():
            jvalue = self.value
        else:
            invalidInputError(sc, 'sc cannot be null to transform a LocalTextSet to a DistributedTextSet')
            jvalue = callZooFunc(self.bigdl_type, 'textSetToDistributed', self.value, sc, partition_num)
        return DistributedTextSet(jvalue=jvalue)

    def to_local(self):
        if False:
            return 10
        '\n        Convert to a LocalTextSet.\n\n        :return: LocalTextSet\n        '
        if self.is_local():
            jvalue = self.value
        else:
            jvalue = callZooFunc(self.bigdl_type, 'textSetToLocal', self.value)
        return LocalTextSet(jvalue=jvalue)

    def get_word_index(self):
        if False:
            print('Hello World!')
        "\n        Get the word_index dictionary of the TextSet.\n        If the TextSet hasn't been transformed from word to index, None will be returned.\n\n        :return: Dictionary {word: id}\n        "
        return callZooFunc(self.bigdl_type, 'textSetGetWordIndex', self.value)

    def save_word_index(self, path):
        if False:
            print('Hello World!')
        '\n        Save the word_index dictionary to text file, which can be used for future inference.\n        Each separate line will be "word id".\n\n        For LocalTextSet, save txt to a local file system.\n        For DistributedTextSet, save txt to a local or distributed file system (such as HDFS).\n\n        :param path: The path to the text file.\n        '
        callZooFunc(self.bigdl_type, 'textSetSaveWordIndex', self.value, path)

    def load_word_index(self, path):
        if False:
            return 10
        '\n        Load the word_index map which was saved after the training, so that this TextSet can\n        directly use this word_index during inference.\n        Each separate line should be "word id".\n\n        Note that after calling `load_word_index`, you do not need to specify any argument when\n        calling `word2idx` in the preprocessing pipeline as now you are using exactly the loaded\n        word_index for transformation.\n\n        For LocalTextSet, load txt from a local file system.\n        For DistributedTextSet, load txt from a local or distributed file system (such as HDFS).\n\n        :return: TextSet with the loaded word_index.\n        '
        jvalue = callZooFunc(self.bigdl_type, 'textSetLoadWordIndex', self.value, path)
        return TextSet(jvalue=jvalue)

    def set_word_index(self, vocab):
        if False:
            return 10
        '\n        Assign a word_index dictionary for this TextSet to use during word2idx.\n        If you load the word_index from the saved file, you are recommended to use `load_word_index`\n        directly.\n\n        :return: TextSet with the word_index set.\n        '
        jvalue = callZooFunc(self.bigdl_type, 'textSetSetWordIndex', self.value, vocab)
        return TextSet(jvalue=jvalue)

    def generate_word_index_map(self, remove_topN=0, max_words_num=-1, min_freq=1, existing_map=None):
        if False:
            return 10
        "\n        Generate word_index map based on sorted word frequencies in descending order.\n        Return the result dictionary, which can also be retrieved by 'get_word_index()'.\n        Make sure you call this after tokenize. Otherwise you will get an error.\n        See word2idx for more details.\n\n        :return: Dictionary {word: id}\n        "
        return callZooFunc(self.bigdl_type, 'textSetGenerateWordIndexMap', self.value, remove_topN, max_words_num, min_freq, existing_map)

    def get_texts(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the text contents of a TextSet.\n\n        :return: List of String for LocalTextSet.\n                 RDD of String for DistributedTextSet.\n        '
        return callZooFunc(self.bigdl_type, 'textSetGetTexts', self.value)

    def get_uris(self):
        if False:
            while True:
                i = 10
        "\n        Get the identifiers of a TextSet.\n        If a text doesn't have a uri, its corresponding position will be None.\n\n        :return: List of String for LocalTextSet.\n                 RDD of String for DistributedTextSet.\n        "
        return callZooFunc(self.bigdl_type, 'textSetGetURIs', self.value)

    def get_labels(self):
        if False:
            return 10
        "\n        Get the labels of a TextSet (if any).\n        If a text doesn't have a label, its corresponding position will be -1.\n\n        :return: List of int for LocalTextSet.\n                 RDD of int for DistributedTextSet.\n        "
        return callZooFunc(self.bigdl_type, 'textSetGetLabels', self.value)

    def get_predicts(self):
        if False:
            return 10
        "\n        Get the prediction results (if any) combined with uris (if any) of a TextSet.\n        If a text doesn't have a uri, its corresponding uri will be None.\n        If a text hasn't been predicted by a model, its corresponding prediction will be None.\n\n        :return: List of (uri, prediction as a list of numpy array) for LocalTextSet.\n                 RDD of (uri, prediction as a list of numpy array) for DistributedTextSet.\n        "
        predicts = callZooFunc(self.bigdl_type, 'textSetGetPredicts', self.value)
        if isinstance(predicts, RDD):
            return predicts.map(lambda predict: (predict[0], _process_predict_result(predict[1])))
        else:
            return [(predict[0], _process_predict_result(predict[1])) for predict in predicts]

    def get_samples(self):
        if False:
            while True:
                i = 10
        "\n        Get the BigDL Sample representations of a TextSet (if any).\n        If a text hasn't been transformed to Sample, its corresponding position will be None.\n\n        :return: List of Sample for LocalTextSet.\n                 RDD of Sample for DistributedTextSet.\n        "
        return callZooFunc(self.bigdl_type, 'textSetGetSamples', self.value)

    def random_split(self, weights):
        if False:
            for i in range(10):
                print('nop')
        '\n        Randomly split into list of TextSet with provided weights.\n        Only available for DistributedTextSet for now.\n\n        :param weights: List of float indicating the split portions.\n        '
        jvalues = callZooFunc(self.bigdl_type, 'textSetRandomSplit', self.value, weights)
        return [TextSet(jvalue=jvalue) for jvalue in list(jvalues)]

    def tokenize(self):
        if False:
            return 10
        '\n        Do tokenization on original text.\n        See Tokenizer for more details.\n\n        :return: TextSet after tokenization.\n        '
        jvalue = callZooFunc(self.bigdl_type, 'textSetTokenize', self.value)
        return TextSet(jvalue=jvalue)

    def normalize(self):
        if False:
            print('Hello World!')
        '\n        Do normalization on tokens.\n        Need to tokenize first.\n        See Normalizer for more details.\n\n        :return: TextSet after normalization.\n        '
        jvalue = callZooFunc(self.bigdl_type, 'textSetNormalize', self.value)
        return TextSet(jvalue=jvalue)

    def word2idx(self, remove_topN=0, max_words_num=-1, min_freq=1, existing_map=None):
        if False:
            i = 10
            return i + 15
        "\n        Map word tokens to indices.\n        Important: Take care that this method behaves a bit differently for training and inference.\n\n        ---------------------------------------Training--------------------------------------------\n        During the training, you need to generate a new word_index dictionary according to the texts\n        you are dealing with. Thus this method will first do the dictionary generation and then\n        convert words to indices based on the generated dictionary.\n\n        You can specify the following arguments which pose some constraints when generating\n        the dictionary.\n        In the result dictionary, index will start from 1 and corresponds to the occurrence\n        frequency of each word sorted in descending order.\n        Here we adopt the convention that index 0 will be reserved for unknown words.\n        After word2idx, you can get the generated word_index dictionary by calling 'get_word_index'.\n        Also, you can call `save_word_index` to save this word_index dictionary to be used in\n        future training.\n\n        :param remove_topN: Non-negative int. Remove the topN words with highest frequencies\n                            in the case where those are treated as stopwords.\n                            Default is 0, namely remove nothing.\n        :param max_words_num: Int. The maximum number of words to be taken into consideration.\n                              Default is -1, namely all words will be considered.\n                              Otherwise, it should be a positive int.\n        :param min_freq: Positive int. Only those words with frequency >= min_freq will be taken\n                         into consideration.\n                         Default is 1, namely all words that occur will be considered.\n        :param existing_map: Existing dictionary of word_index if any.\n                             Default is None and in this case a new dictionary with index starting\n                             from 1 will be generated.\n                             If not None, then the generated dictionary will preserve the word_index\n                             in existing_map and assign subsequent indices to new words.\n\n        ---------------------------------------Inference--------------------------------------------\n        During the inference, you are supposed to use exactly the same word_index dictionary as in\n        the training stage instead of generating a new one.\n        Thus please be aware that you do not need to specify any of the above arguments.\n        You need to call `load_word_index` or `set_word_index` beforehand for dictionary loading.\n\n        Need to tokenize first.\n        See WordIndexer for more details.\n\n        :return: TextSet after word2idx.\n        "
        jvalue = callZooFunc(self.bigdl_type, 'textSetWord2idx', self.value, remove_topN, max_words_num, min_freq, existing_map)
        return TextSet(jvalue=jvalue)

    def shape_sequence(self, len, trunc_mode='pre', pad_element=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Shape the sequence of indices to a fixed length.\n        Need to word2idx first.\n        See SequenceShaper for more details.\n\n        :return: TextSet after sequence shaping.\n        '
        invalidInputError(isinstance(pad_element, int), 'pad_element should be an int')
        jvalue = callZooFunc(self.bigdl_type, 'textSetShapeSequence', self.value, len, trunc_mode, pad_element)
        return TextSet(jvalue=jvalue)

    def generate_sample(self):
        if False:
            i = 10
            return i + 15
        '\n        Generate BigDL Sample.\n        Need to word2idx first.\n        See TextFeatureToSample for more details.\n\n        :return: TextSet with Samples.\n        '
        jvalue = callZooFunc(self.bigdl_type, 'textSetGenerateSample', self.value)
        return TextSet(jvalue=jvalue)

    def transform(self, transformer):
        if False:
            while True:
                i = 10
        return TextSet(callZooFunc(self.bigdl_type, 'transformTextSet', transformer, self.value), self.bigdl_type)

    @classmethod
    def read(cls, path, sc=None, min_partitions=1, bigdl_type='float'):
        if False:
            print('Hello World!')
        '\n        Read text files with labels from a directory.\n        The folder structure is expected to be the following:\n        path\n          |dir1 - text1, text2, ...\n          |dir2 - text1, text2, ...\n          |dir3 - text1, text2, ...\n        Under the target path, there ought to be N subdirectories (dir1 to dirN). Each\n        subdirectory represents a category and contains all texts that belong to such\n        category. Each category will be a given a label according to its position in the\n        ascending order sorted among all subdirectories.\n        All texts will be given a label according to the subdirectory where it is located.\n        Labels start from 0.\n\n        :param path: The folder path to texts. Local or distributed file system (such as HDFS)\n                     are supported. If you want to read from a distributed file system, sc\n                     needs to be specified.\n        :param sc: An instance of SparkContext.\n                   If specified, texts will be read as a DistributedTextSet.\n                   Default is None and in this case texts will be read as a LocalTextSet.\n        :param min_partitions: Int. A suggestion value of the minimal partition number for input\n                               texts. Only need to specify this when sc is not None. Default is 1.\n\n        :return: TextSet.\n        '
        jvalue = callZooFunc(bigdl_type, 'readTextSet', path, sc, min_partitions)
        return TextSet(jvalue=jvalue)

    @classmethod
    def read_csv(cls, path, sc=None, min_partitions=1, bigdl_type='float'):
        if False:
            print('Hello World!')
        '\n        Read texts with id from csv file.\n        Each record is supposed to contain the following two fields in order:\n        id(string) and text(string).\n        Note that the csv file should be without header.\n\n        :param path: The path to the csv file. Local or distributed file system (such as HDFS)\n                     are supported. If you want to read from a distributed file system, sc\n                     needs to be specified.\n        :param sc: An instance of SparkContext.\n                   If specified, texts will be read as a DistributedTextSet.\n                   Default is None and in this case texts will be read as a LocalTextSet.\n        :param min_partitions: Int. A suggestion value of the minimal partition number for input\n                               texts. Only need to specify this when sc is not None. Default is 1.\n\n        :return: TextSet.\n        '
        jvalue = callZooFunc(bigdl_type, 'textSetReadCSV', path, sc, min_partitions)
        return TextSet(jvalue=jvalue)

    @classmethod
    def read_parquet(cls, path, sc, bigdl_type='float'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read texts with id from parquet file.\n        Schema should be the following:\n        "id"(string) and "text"(string).\n\n        :param path: The path to the parquet file.\n        :param sc: An instance of SparkContext.\n\n        :return: DistributedTextSet.\n        '
        jvalue = callZooFunc(bigdl_type, 'textSetReadParquet', path, sc)
        return DistributedTextSet(jvalue=jvalue)

    @classmethod
    def from_relation_pairs(cls, relations, corpus1, corpus2, bigdl_type='float'):
        if False:
            return 10
        '\n        Used to generate a TextSet for pairwise training.\n\n        This method does the following:\n        1. Generate all RelationPairs: (id1, id2Positive, id2Negative) from Relations.\n        2. Join RelationPairs with corpus to transform id to indexedTokens.\n        Note: Make sure that the corpus has been transformed by SequenceShaper and WordIndexer.\n        3. For each pair, generate a TextFeature having Sample with:\n        - feature of shape (2, text1Length + text2Length).\n        - label of value [1 0] as the positive relation is placed before the negative one.\n\n        :param relations: List or RDD of Relation.\n        :param corpus1: TextSet that contains all id1 in relations. For each TextFeature in corpus1,\n                        text must have been transformed to indexedTokens of the same length.\n        :param corpus2: TextSet that contains all id2 in relations. For each TextFeature in corpus2,\n                        text must have been transformed to indexedTokens of the same length.\n        Note that if relations is a list, then corpus1 and corpus2 must both be LocalTextSet.\n        If relations is RDD, then corpus1 and corpus2 must both be DistributedTextSet.\n\n        :return: TextSet.\n        '
        if isinstance(relations, RDD):
            relations = relations.map(lambda x: x.to_tuple())
        elif isinstance(relations, list):
            relations = [relation.to_tuple() for relation in relations]
        else:
            invalidInputError(False, 'relations should be RDD or list of Relation')
        jvalue = callZooFunc(bigdl_type, 'textSetFromRelationPairs', relations, corpus1, corpus2)
        return TextSet(jvalue=jvalue)

    @classmethod
    def from_relation_lists(cls, relations, corpus1, corpus2, bigdl_type='float'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Used to generate a TextSet for ranking.\n\n        This method does the following:\n        1. For each id1 in relations, find the list of id2 with corresponding label that\n        comes together with id1.\n        In other words, group relations by id1.\n        2. Join with corpus to transform each id to indexedTokens.\n        Note: Make sure that the corpus has been transformed by SequenceShaper and WordIndexer.\n        3. For each list, generate a TextFeature having Sample with:\n        - feature of shape (list_length, text1_length + text2_length).\n        - label of shape (list_length, 1).\n\n        :param relations: List or RDD of Relation.\n        :param corpus1: TextSet that contains all id1 in relations. For each TextFeature in corpus1,\n                        text must have been transformed to indexedTokens of the same length.\n        :param corpus2: TextSet that contains all id2 in relations. For each TextFeature in corpus2,\n                        text must have been transformed to indexedTokens of the same length.\n        Note that if relations is a list, then corpus1 and corpus2 must both be LocalTextSet.\n        If relations is RDD, then corpus1 and corpus2 must both be DistributedTextSet.\n\n        :return: TextSet.\n        '
        if isinstance(relations, RDD):
            relations = relations.map(lambda x: x.to_tuple())
        elif isinstance(relations, list):
            relations = [relation.to_tuple() for relation in relations]
        else:
            invalidInputError(False, 'relations should be RDD or list of Relation')
        jvalue = callZooFunc(bigdl_type, 'textSetFromRelationLists', relations, corpus1, corpus2)
        return TextSet(jvalue=jvalue)

class LocalTextSet(TextSet):
    """
    LocalTextSet is comprised of lists.
    """

    def __init__(self, texts=None, labels=None, jvalue=None, bigdl_type='float'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Create a LocalTextSet using texts and labels.\n\n        # Arguments:\n        texts: List of String. Each element is the content of a text.\n        labels: List of int or None if texts don't have labels.\n        "
        if texts is not None:
            invalidInputError(all((isinstance(text, six.string_types) for text in texts)), 'texts for LocalTextSet should be list of string')
        if labels is not None:
            labels = [int(label) for label in labels]
        super(LocalTextSet, self).__init__(jvalue, bigdl_type, texts, labels)

class DistributedTextSet(TextSet):
    """
    DistributedTextSet is comprised of RDDs.
    """

    def __init__(self, texts=None, labels=None, jvalue=None, bigdl_type='float'):
        if False:
            i = 10
            return i + 15
        "\n        Create a DistributedTextSet using texts and labels.\n\n        # Arguments:\n        texts: RDD of String. Each element is the content of a text.\n        labels: RDD of int or None if texts don't have labels.\n        "
        if texts is not None:
            invalidInputError(isinstance(texts, RDD), 'texts for DistributedTextSet should be RDD of String')
        if labels is not None:
            invalidInputError(isinstance(labels, RDD), 'labels for DistributedTextSet should be RDD of int')
            labels = labels.map(lambda x: int(x))
        super(DistributedTextSet, self).__init__(jvalue, bigdl_type, texts, labels)

def _process_predict_result(predict):
    if False:
        print('Hello World!')
    if predict is not None:
        return [res.to_ndarray() for res in predict]
    else:
        return None