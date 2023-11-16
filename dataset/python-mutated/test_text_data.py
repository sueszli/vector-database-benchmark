"""Test for the TextData object"""
import numpy as np
import pandas as pd
from hamcrest import assert_that, calling, contains_exactly, equal_to, raises
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp.text_data import TextData

def test_text_data_init():
    if False:
        print('Hello World!')
    'Test the TextData object initialization'
    text_data = TextData(['Hello world'])
    assert_that(text_data.text, contains_exactly('Hello world'))

def test_init_no_text():
    if False:
        print('Hello World!')
    'Test the TextData object when no text is provided'
    assert_that(calling(TextData).with_args([1]), raises(DeepchecksValueError, 'raw_text must be a Sequence of strings'))

def test_init_mismatched_task_type():
    if False:
        return 10
    'Test the TextData object when the task type does not match the label format'
    label = [1, 2, 3]
    text = ['a', 'b', 'c']
    assert_that(calling(TextData).with_args(raw_text=text, label=label, task_type='token_classification'), raises(DeepchecksValueError, 'tokenized_text must be provided for token_classification task type'))
    label = [['PER', 'ORG', 'ORG', 'GEO'], [], []]
    assert_that(calling(TextData).with_args(raw_text=text, label=label, task_type='text_classification'), raises(DeepchecksValueError, 'multilabel was identified. It must be a Sequence of Sequences of 0 or 1.'))

def test_init_no_labels():
    if False:
        return 10
    'Test the TextData object when no labels are provided'
    text = ['I think therefore I am', 'I am therefore I think', 'I am']
    label = [None, None, None]
    text_data = TextData(raw_text=text, label=label, task_type='text_classification')
    assert_that(text_data.has_label(), equal_to(False))
    assert_that(text_data.is_multi_label_classification(), equal_to(False))
    label = [np.nan, pd.NA, np.nan]
    text_data = TextData(raw_text=text, label=label, task_type='text_classification')
    assert_that(text_data.has_label(), equal_to(False))
    assert_that(text_data.is_multi_label_classification(), equal_to(False))

def test_wrong_token_label_format():
    if False:
        for i in range(10):
            print('nop')
    tokenized_text = [['a'], ['b', 'b', 'b'], ['c', 'c', 'c', 'c']]
    label_structure_error = 'label must be a Sequence of Sequences of either strings or integers'
    label = [['B-PER'], ['B-PER', 'B-GEO', 'B-GEO'], ['B-PER', 'B-GEO', 'B-GEO', 'B-GEO']]
    _ = TextData(tokenized_text=tokenized_text, label=label, task_type='token_classification')
    label = 'PER'
    assert_that(calling(TextData).with_args(tokenized_text=tokenized_text, label=label, task_type='token_classification'), raises(DeepchecksValueError, 'label must be a Sequence'))
    label = [3, 3, 3]
    assert_that(calling(TextData).with_args(tokenized_text=tokenized_text, label=label, task_type='token_classification'), raises(DeepchecksValueError, label_structure_error))
    label = [['B-PER'], 1, ['B-PER', 'B-GEO', 'B-GEO', 'B-GEO']]
    assert_that(calling(TextData).with_args(tokenized_text=tokenized_text, label=label, task_type='token_classification'), raises(DeepchecksValueError, label_structure_error))
    label = [['B-PER'], ['B-PER', 'B-GEO', 'B-GEO'], ['B-PER', 'B-GEO', 'B-GEO']]
    assert_that(calling(TextData).with_args(tokenized_text=tokenized_text, label=label, task_type='token_classification'), raises(DeepchecksValueError, 'label must be the same length as tokenized_text. However, for sample index 2 received token list of length 4 and label list of length 3'))

def test_text_data_initialization_with_incorrect_type_of_metadata():
    if False:
        while True:
            i = 10
    text = ['a', 'b b b', 'c c c c']
    metadata = {'first': [1, 2, 3], 'second': [4, 5, 6]}
    _ = TextData(raw_text=text, metadata=pd.DataFrame(metadata), task_type='text_classification')
    assert_that(calling(TextData).with_args(raw_text=text, metadata=metadata, task_type='text_classification'), raises(DeepchecksValueError, "Metadata type <class 'dict'> is not supported, must be a pandas DataFrame"))

def test_head_functionality():
    if False:
        while True:
            i = 10
    text = ['a', 'b b b', 'c c c c']
    metadata = {'first': [1, 2, 3], 'second': [4, 5, 6]}
    label = ['PER', 'ORG', 'GEO']
    dataset = TextData(raw_text=text, metadata=pd.DataFrame(metadata), task_type='text_classification', label=label)
    result = dataset.head(n_samples=2)
    assert_that(len(result), equal_to(2))
    assert_that(sorted(result.columns), contains_exactly('first', 'label', 'second', 'text'))
    assert_that(list(result.index), contains_exactly(0, 1))

def test_label_for_display():
    if False:
        print('Hello World!')
    text = ['a', 'b b b', 'c c c c']
    single_label = ['PER', 'ORG', 'GEO']
    multi_label = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
    dataset = TextData(raw_text=text, task_type='text_classification', label=single_label)
    result = dataset.label_for_display()
    assert_that(len(result), equal_to(3))
    assert_that(result, contains_exactly('PER', 'ORG', 'GEO'))
    dataset = TextData(raw_text=text, task_type='text_classification', label=multi_label)
    result = dataset.label_for_display()
    assert_that(len(result), equal_to(3))
    assert_that(result[0], contains_exactly(0, 2))
    result = dataset.label_for_display(model_classes=['PER', 'ORG', 'GEO'])
    assert_that(len(result), equal_to(3))
    assert_that(result[0], contains_exactly('PER', 'GEO'))

def test_properties(text_classification_dataset_mock):
    if False:
        while True:
            i = 10
    dataset = text_classification_dataset_mock
    assert_that(dataset._properties, equal_to(None))
    dataset.calculate_builtin_properties(include_long_calculation_properties=False)
    properties = dataset.properties
    assert_that(properties.shape[0], equal_to(3))
    assert_that(properties.shape[1], equal_to(11))
    assert_that(properties.columns, contains_exactly('Text Length', 'Average Word Length', 'Max Word Length', '% Special Characters', '% Punctuation', 'Language', 'Sentiment', 'Subjectivity', 'Average Words Per Sentence', 'Reading Ease', 'Lexical Density'))
    assert_that(properties.iloc[0].values, contains_exactly(22, 3.6, 9, 0.0, 0.0, 'en', 0.0, 0.0, 5.0, 100.24, 80.0))

def test_embeddings():
    if False:
        i = 10
        return i + 15
    ds = TextData(['my name is inigo montoya', 'you killed my father', 'prepare to die'])
    ds.calculate_builtin_embeddings()
    assert_that(ds.embeddings.shape, equal_to((3, 384)))

def test_set_embeddings(text_classification_dataset_mock):
    if False:
        return 10
    dataset = text_classification_dataset_mock
    embeddings = pd.DataFrame({'0': [1, 2, 3], '1': [4, 5, 6]})
    assert_that(dataset._embeddings, equal_to(None))
    dataset.set_embeddings(embeddings)
    assert_that((dataset.embeddings != embeddings).sum().sum(), equal_to(0))
    dataset._embeddings = None
    embeddings = np.array([[1, 2, 3], [4, 5, 6]]).T
    dataset.set_embeddings(embeddings)
    assert_that((dataset.embeddings != embeddings).sum().sum(), equal_to(0))

def test_set_metadata(text_classification_dataset_mock):
    if False:
        for i in range(10):
            print('nop')
    dataset = text_classification_dataset_mock
    metadata = pd.DataFrame({'first': [1, 2, 3], 'second': [4, 5, 6]})
    assert_that(dataset._metadata, equal_to(None))
    assert_that(dataset._cat_metadata, equal_to(None))
    dataset.set_metadata(metadata, categorical_metadata=[])
    assert_that((dataset.metadata != metadata).sum().sum(), equal_to(0))
    assert_that(dataset.categorical_metadata, equal_to([]))

def test_set_metadata_with_categorical_columns(text_classification_dataset_mock):
    if False:
        print('Hello World!')
    dataset = text_classification_dataset_mock
    metadata = pd.DataFrame({'first': [1, 2, 3], 'second': [4, 5, 6]})
    assert_that(dataset._metadata, equal_to(None))
    assert_that(dataset._cat_metadata, equal_to(None))
    dataset.set_metadata(metadata, categorical_metadata=['second'])
    assert_that((dataset.metadata != metadata).sum().sum(), equal_to(0))
    assert_that(dataset.categorical_metadata, equal_to(['second']))

def test_set_metadata_with_an_incorrect_list_of_categorical_columns(text_classification_dataset_mock):
    if False:
        while True:
            i = 10
    dataset = text_classification_dataset_mock
    metadata = pd.DataFrame({'first': [1, 2, 3], 'second': [4, 5, 6]})
    assert_that(dataset._metadata, equal_to(None))
    assert_that(dataset._cat_metadata, equal_to(None))
    assert_that(calling(dataset.set_metadata).with_args(metadata, categorical_metadata=['foo']), raises(DeepchecksValueError, "The following columns does not exist in Metadata - \\['foo'\\]"))

def test_load_metadata(text_classification_dataset_mock):
    if False:
        return 10
    dataset = text_classification_dataset_mock
    metadata = pd.DataFrame({'first': [1, 2, 3], 'second': [4, 5, 6]})
    assert_that(dataset._metadata, equal_to(None))
    assert_that(dataset._cat_metadata, equal_to(None))
    metadata.to_csv('metadata.csv', index=False)
    loaded_metadata = pd.read_csv('metadata.csv')
    assert_that((loaded_metadata != metadata).sum().sum(), equal_to(0))
    dataset.set_metadata(loaded_metadata)
    assert_that((dataset.metadata != metadata).sum().sum(), equal_to(0))

def test_set_properties(text_classification_dataset_mock):
    if False:
        i = 10
        return i + 15
    dataset = text_classification_dataset_mock
    properties = pd.DataFrame({'text_length': [1, 2, 3], 'average_word_length': [4, 5, 6]})
    assert_that(dataset._properties, equal_to(None))
    assert_that(dataset._cat_properties, equal_to(None))
    dataset.set_properties(properties, categorical_properties=[])
    assert_that(dataset.categorical_properties, equal_to([]))
    assert_that((dataset.properties != properties).sum().sum(), equal_to(0))
    dataset._properties = None
    dataset._cat_properties = None

def test_set_properties_with_builtin(text_classification_dataset_mock):
    if False:
        print('Hello World!')
    dataset = text_classification_dataset_mock
    properties = pd.DataFrame({'Language': ['en', 'en', 'es'], 'Average Word Length': [4, 5, 6]})
    assert_that(dataset._properties, equal_to(None))
    assert_that(dataset._cat_properties, equal_to(None))
    dataset.set_properties(properties)
    assert_that(dataset.categorical_properties, equal_to(['Language']))
    assert_that((dataset.properties != properties).sum().sum(), equal_to(0))

def test_set_properties_with_an_incorrect_list_of_categorical_columns(text_classification_dataset_mock):
    if False:
        print('Hello World!')
    dataset = text_classification_dataset_mock
    properties = pd.DataFrame({'text_length': [1, 2, 3], 'average_word_length': [4, 5, 6]})
    assert_that(calling(dataset.set_properties).with_args(properties, categorical_properties=['foo']), raises(DeepchecksValueError, "The following columns does not exist in Properties - \\['foo'\\]"))

def test_set_properties_with_categorical_columns(text_classification_dataset_mock):
    if False:
        return 10
    dataset = text_classification_dataset_mock
    properties = pd.DataFrame({'unknown_property': ['foo', 'foo', 'bar']})
    assert_that(dataset._properties, equal_to(None))
    assert_that(dataset._cat_properties, equal_to(None))
    dataset.set_properties(properties)
    assert_that(dataset.categorical_properties, equal_to(['unknown_property']))

def test_save_and_load_properties(text_classification_dataset_mock):
    if False:
        print('Hello World!')
    dataset = text_classification_dataset_mock
    properties = pd.DataFrame({'text_length': [1, 2, 3], 'average_word_length': [4, 5, 6]})
    assert_that(dataset._properties, equal_to(None))
    assert_that(dataset._cat_properties, equal_to(None))
    dataset.set_properties(properties, categorical_properties=[])
    dataset.save_properties('test_properties.csv')
    properties_loaded = pd.read_csv('test_properties.csv')
    assert_that((properties_loaded != properties).sum().sum(), equal_to(0))
    dataset._properties = None
    dataset.set_properties('test_properties.csv')
    assert_that((dataset.properties != properties).sum().sum(), equal_to(0))

def test_mixed_builtin_and_mixed_properties(text_classification_dataset_mock):
    if False:
        i = 10
        return i + 15
    dataset = text_classification_dataset_mock
    properties = pd.DataFrame({'custom': [1, 2, 3], 'Average Word Length': [4, 5, 6]})
    dataset.set_properties(properties, categorical_properties=[])
    assert_that(dataset.categorical_properties, equal_to([]))
    dataset = text_classification_dataset_mock
    properties = pd.DataFrame({'custom': [1, 2, 3], 'Language': ['en', 'en', 'es']})
    dataset.set_properties(properties, categorical_properties=[])
    assert_that(dataset.categorical_properties, equal_to(['Language']))

def test_describe_with_properties(text_multilabel_classification_dataset_mock, tweet_emotion_train_test_textdata):
    if False:
        for i in range(10):
            print('nop')
    dataset_without_properties = text_multilabel_classification_dataset_mock
    (dataset_with_properties, _) = tweet_emotion_train_test_textdata
    figure_without_properties = dataset_without_properties.describe(n_properties_to_show=8)
    figure_with_properties_one = dataset_with_properties.describe(n_properties_to_show=3)
    figure_with_properties_two = dataset_with_properties.describe(properties_to_show=['Text Length', 'Language'])
    assert_that(calling(dataset_without_properties.describe).with_args(properties_to_show=['Property One']), raises(DeepchecksValueError, 'No properties exist!'))
    assert_that(len(figure_without_properties.data), equal_to(2))
    assert_that(len(figure_without_properties.layout.annotations), equal_to(1))
    assert_that(figure_without_properties.data[0].type, equal_to('pie'))
    assert_that(figure_without_properties.data[1].type, equal_to('table'))
    assert_that(len(figure_with_properties_one.data), equal_to(5))
    assert_that(len(figure_with_properties_one.layout.annotations), equal_to(16))
    assert_that(figure_with_properties_one.data[0].type, equal_to('pie'))
    assert_that(figure_with_properties_one.data[1].type, equal_to('table'))
    assert_that(figure_with_properties_one.data[2].type, equal_to('scatter'))
    assert_that(figure_with_properties_one.data[3].type, equal_to('scatter'))
    assert_that(figure_with_properties_one.data[4].type, equal_to('scatter'))
    assert_that(len(figure_with_properties_two.data), equal_to(4))
    assert_that(len(figure_with_properties_two.layout.annotations), equal_to(7))
    assert_that(figure_with_properties_two.data[0].type, equal_to('pie'))
    assert_that(figure_with_properties_two.data[1].type, equal_to('table'))
    assert_that(figure_with_properties_two.data[2].type, equal_to('scatter'))
    assert_that(figure_with_properties_two.data[3].type, equal_to('bar'))

def test_describe_with_multi_label_dataset(text_multilabel_classification_dataset_mock):
    if False:
        for i in range(10):
            print('nop')
    dataset = text_multilabel_classification_dataset_mock
    figure = dataset.describe()
    assert_that(len(figure.data), equal_to(2))
    assert_that(len(figure.layout.annotations), equal_to(1))
    assert_that(figure.data[0].type, equal_to('pie'))
    assert_that(figure.data[1].type, equal_to('table'))

def test_describe_with_single_label_dataset(tweet_emotion_train_test_textdata):
    if False:
        return 10
    (dataset, _) = tweet_emotion_train_test_textdata
    figure = dataset.describe(n_properties_to_show=2)
    assert_that(len(figure.data), equal_to(4))
    assert_that(len(figure.layout.annotations), equal_to(11))
    assert_that(figure.data[0].type, equal_to('pie'))
    assert_that(figure.data[1].type, equal_to('table'))
    assert_that(figure.data[2].type, equal_to('scatter'))
    assert_that(figure.data[3].type, equal_to('scatter'))

def test_describe_with_token_classification_dataset(text_token_classification_dataset_mock):
    if False:
        return 10
    dataset = text_token_classification_dataset_mock
    figure = dataset.describe()
    assert_that(len(figure.data), equal_to(2))
    assert_that(len(figure.layout.annotations), equal_to(1))
    assert_that(figure.data[0].type, equal_to('pie'))
    assert_that(figure.data[1].type, equal_to('table'))