import logging
from unittest.mock import patch
from pathlib import Path
import pytest
from canals.errors import PipelineRuntimeError
from langdetect import LangDetectException
from haystack.preview.components.file_converters.txt import TextFileToDocument

class TestTextfileToDocument:

    @pytest.mark.unit
    def test_run(self, preview_samples_path):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if the component runs correctly.\n        '
        paths = [preview_samples_path / 'txt' / 'doc_1.txt', preview_samples_path / 'txt' / 'doc_2.txt']
        converter = TextFileToDocument()
        output = converter.run(paths=paths)
        docs = output['documents']
        assert len(docs) == 2
        assert docs[0].content == 'Some text for testing.\nTwo lines in here.'
        assert docs[1].content == 'This is a test line.\n123 456 789\n987 654 321.'
        assert docs[0].meta['file_path'] == str(paths[0])
        assert docs[1].meta['file_path'] == str(paths[1])

    @pytest.mark.unit
    def test_run_warning_for_invalid_language(self, preview_samples_path, caplog):
        if False:
            print('Hello World!')
        file_path = preview_samples_path / 'txt' / 'doc_1.txt'
        converter = TextFileToDocument()
        with patch('haystack.preview.components.file_converters.txt.langdetect.detect', return_value='en'), caplog.at_level(logging.WARNING):
            output = converter.run(paths=[file_path], valid_languages=['de'])
            assert f"Text from file {file_path} is not in one of the valid languages: ['de']. The file may have been decoded incorrectly." in caplog.text
        docs = output['documents']
        assert len(docs) == 1
        assert docs[0].content == 'Some text for testing.\nTwo lines in here.'

    @pytest.mark.unit
    def test_run_error_handling(self, preview_samples_path, caplog):
        if False:
            while True:
                i = 10
        '\n        Test if the component correctly handles errors.\n        '
        paths = [preview_samples_path / 'txt' / 'doc_1.txt', 'non_existing_file.txt']
        converter = TextFileToDocument()
        with caplog.at_level(logging.WARNING):
            output = converter.run(paths=paths)
            assert 'Could not read file non_existing_file.txt. Skipping it. Error message: File at path non_existing_file.txt does not exist.' in caplog.text
        docs = output['documents']
        assert len(docs) == 1
        assert docs[0].meta['file_path'] == str(paths[0])

    @pytest.mark.unit
    def test_prepare_metadata_no_metadata(self):
        if False:
            return 10
        '\n        Test if the metadata is correctly prepared when no custom metadata is provided.\n        '
        converter = TextFileToDocument()
        meta = converter._prepare_metadata(metadata=None, paths=['data/sample_path_1.txt', Path('data/sample_path_2.txt')])
        assert len(meta) == 2
        assert meta[0]['file_path'] == 'data/sample_path_1.txt'
        assert meta[1]['file_path'] == str(Path('data/sample_path_2.txt'))

    @pytest.mark.unit
    def test_prepare_metadata_single_dict(self):
        if False:
            return 10
        '\n        Test if the metadata is correctly prepared when a single dict is provided.\n        '
        converter = TextFileToDocument()
        meta = converter._prepare_metadata(metadata={'name': 'test'}, paths=['data/sample_path_1.txt', Path('data/sample_path_2.txt')])
        assert len(meta) == 2
        assert meta[0]['file_path'] == 'data/sample_path_1.txt'
        assert meta[1]['file_path'] == str(Path('data/sample_path_2.txt'))
        assert meta[0]['name'] == 'test'
        assert meta[1]['name'] == 'test'

    @pytest.mark.unit
    def test_prepare_metadata_list_of_dicts(self):
        if False:
            print('Hello World!')
        '\n        Test if the metadata is correctly prepared when a list of dicts is provided.\n        '
        converter = TextFileToDocument()
        meta = converter._prepare_metadata(metadata=[{'name': 'test1'}, {'name': 'test2'}], paths=['data/sample_path_1.txt', Path('data/sample_path_2.txt')])
        assert len(meta) == 2
        assert meta[0]['file_path'] == 'data/sample_path_1.txt'
        assert meta[1]['file_path'] == str(Path('data/sample_path_2.txt'))
        assert meta[0]['name'] == 'test1'
        assert meta[1]['name'] == 'test2'

    @pytest.mark.unit
    def test_prepare_metadata_unmatching_list_len(self):
        if False:
            print('Hello World!')
        '\n        Test if an error is raised when the number of metadata dicts is not equal to the number of\n        file paths.\n        '
        converter = TextFileToDocument()
        with pytest.raises(PipelineRuntimeError, match='The number of metadata entries must match the number of paths if metadata is a list.'):
            converter._prepare_metadata(metadata=[{'name': 'test1'}, {'name': 'test2'}], paths=['data/sample_path_1.txt', Path('data/sample_path_2.txt'), 'data/sample_path_3.txt'])

    @pytest.mark.unit
    def test_read_and_clean_file(self, preview_samples_path):
        if False:
            print('Hello World!')
        '\n        Test if the file is correctly read.\n        '
        file_path = preview_samples_path / 'txt' / 'doc_1.txt'
        converter = TextFileToDocument()
        text = converter._read_and_clean_file(path=file_path, encoding='utf-8', remove_numeric_tables=False)
        assert text == 'Some text for testing.\nTwo lines in here.'

    @pytest.mark.unit
    def test_read_and_clean_file_non_existing_file(self):
        if False:
            while True:
                i = 10
        '\n        Test if an error is raised when the file does not exist.\n        '
        converter = TextFileToDocument()
        file_path = 'non_existing_file.txt'
        with pytest.raises(PipelineRuntimeError, match=f'File at path {file_path} does not exist.'):
            converter._read_and_clean_file(path=file_path, encoding='utf-8', remove_numeric_tables=False)

    @pytest.mark.unit
    def test_read_and_clean_file_remove_numeric_tables(self, preview_samples_path):
        if False:
            while True:
                i = 10
        '\n        Test if the file is correctly read and numeric tables are removed.\n        '
        file_path = preview_samples_path / 'txt' / 'doc_2.txt'
        converter = TextFileToDocument()
        text = converter._read_and_clean_file(path=file_path, encoding='utf-8', remove_numeric_tables=True)
        assert text == 'This is a test line.\n987 654 321.'

    @pytest.mark.unit
    def test_clean_page_without_remove_numeric_tables(self):
        if False:
            print('Hello World!')
        '\n        Test if the page is not changed when remove_numeric_tables is False.\n        '
        converter = TextFileToDocument()
        page = 'This is a test line.\n123 456 789'
        cleaned_page = converter._clean_page(page=page, remove_numeric_tables=False)
        assert cleaned_page == page

    @pytest.mark.unit
    def test_clean_page_with_remove_numeric_tables(self):
        if False:
            i = 10
            return i + 15
        '\n        Test if the page is correctly cleaned when remove_numeric_tables is True.\n        '
        converter = TextFileToDocument()
        page = 'This is a test line.\n123 456 789'
        cleaned_page = converter._clean_page(page=page, remove_numeric_tables=True)
        assert cleaned_page == 'This is a test line.'

    @pytest.mark.unit
    def test_is_numeric_row_only_numbers(self):
        if False:
            return 10
        '\n        Test if the line is correctly identified as a numeric row when it only contains numbers.\n        '
        converter = TextFileToDocument()
        line = '123 456 789'
        assert converter._is_numeric_row(line=line)

    @pytest.mark.unit
    def test_is_numeric_row_only_text(self):
        if False:
            print('Hello World!')
        '\n        Test if the line is correctly identified as a non-numeric row when it only contains text.\n        '
        converter = TextFileToDocument()
        line = 'This is a test line.'
        assert not converter._is_numeric_row(line=line)

    @pytest.mark.unit
    def test_is_numeric_row_only_numbers_with_period(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if the line is correctly identified as a non-numeric row when it only contains numbers and a period at\n        the end.\n        '
        converter = TextFileToDocument()
        line = '123 456 789.'
        assert not converter._is_numeric_row(line=line)

    @pytest.mark.unit
    def test_is_numeric_row_more_numbers_than_text(self):
        if False:
            print('Hello World!')
        '\n        Test if the line is correctly identified as a numeric row when it consists of more than 40% of numbers than.\n        '
        converter = TextFileToDocument()
        line = '123 456 789 This is a test'
        assert converter._is_numeric_row(line=line)

    @pytest.mark.unit
    def test_is_numeric_row_less_numbers_than_text(self):
        if False:
            i = 10
            return i + 15
        '\n        Test if the line is correctly identified as a non-numeric row when it consists of less than 40% of numbers than.\n        '
        converter = TextFileToDocument()
        line = '123 456 789 This is a test line'
        assert not converter._is_numeric_row(line=line)

    @pytest.mark.unit
    def test_is_numeric_row_words_consist_of_numbers_and_text(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if the line is correctly identified as a numeric row when the words consist of numbers and text.\n        '
        converter = TextFileToDocument()
        line = '123eur 456usd'
        assert converter._is_numeric_row(line=line)

    @pytest.mark.unit
    def test_validate_language(self):
        if False:
            return 10
        '\n        Test if the language is correctly validated.\n        '
        converter = TextFileToDocument()
        with patch('haystack.preview.components.file_converters.txt.langdetect.detect', return_value='en'):
            assert converter._validate_language(text='This is an english text.', valid_languages=['en'])
            assert not converter._validate_language(text='This is an english text.', valid_languages=['de'])

    @pytest.mark.unit
    def test_validate_language_no_languages_specified(self):
        if False:
            print('Hello World!')
        '\n        Test if _validate_languages returns True when no languages are specified.\n        '
        converter = TextFileToDocument()
        assert converter._validate_language(text='This is an english test.', valid_languages=[])

    @pytest.mark.unit
    def test_validate_language_lang_detect_exception(self):
        if False:
            while True:
                i = 10
        '\n        Test if _validate_languages returns False when langdetect throws an exception.\n        '
        converter = TextFileToDocument()
        with patch('haystack.preview.components.file_converters.txt.langdetect.detect', side_effect=LangDetectException(code=0, message='Test')):
            assert not converter._validate_language(text='This is an english text.', valid_languages=['en'])