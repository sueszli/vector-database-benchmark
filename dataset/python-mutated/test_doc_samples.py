import doctest
import logging
import os
import unittest
from pathlib import Path
from typing import List, Union
import transformers
from transformers.testing_utils import require_tf, require_torch, slow
logger = logging.getLogger()

@unittest.skip('Temporarily disable the doc tests.')
@require_torch
@require_tf
@slow
class TestCodeExamples(unittest.TestCase):

    def analyze_directory(self, directory: Path, identifier: Union[str, None]=None, ignore_files: Union[List[str], None]=None, n_identifier: Union[str, List[str], None]=None, only_modules: bool=True):
        if False:
            return 10
        '\n        Runs through the specific directory, looking for the files identified with `identifier`. Executes\n        the doctests in those files\n\n        Args:\n            directory (`Path`): Directory containing the files\n            identifier (`str`): Will parse files containing this\n            ignore_files (`List[str]`): List of files to skip\n            n_identifier (`str` or `List[str]`): Will not parse files containing this/these identifiers.\n            only_modules (`bool`): Whether to only analyze modules\n        '
        files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
        if identifier is not None:
            files = [file for file in files if identifier in file]
        if n_identifier is not None:
            if isinstance(n_identifier, List):
                for n_ in n_identifier:
                    files = [file for file in files if n_ not in file]
            else:
                files = [file for file in files if n_identifier not in file]
        ignore_files = ignore_files or []
        ignore_files.append('__init__.py')
        files = [file for file in files if file not in ignore_files]
        for file in files:
            print('Testing', file)
            if only_modules:
                module_identifier = file.split('.')[0]
                try:
                    module_identifier = getattr(transformers, module_identifier)
                    suite = doctest.DocTestSuite(module_identifier)
                    result = unittest.TextTestRunner().run(suite)
                    self.assertIs(len(result.failures), 0)
                except AttributeError:
                    logger.info(f'{module_identifier} is not a module.')
            else:
                result = doctest.testfile(str('..' / directory / file), optionflags=doctest.ELLIPSIS)
                self.assertIs(result.failed, 0)

    def test_modeling_examples(self):
        if False:
            print('Hello World!')
        transformers_directory = Path('src/transformers')
        files = 'modeling'
        ignore_files = ['modeling_ctrl.py', 'modeling_tf_ctrl.py']
        self.analyze_directory(transformers_directory, identifier=files, ignore_files=ignore_files)

    def test_tokenization_examples(self):
        if False:
            i = 10
            return i + 15
        transformers_directory = Path('src/transformers')
        files = 'tokenization'
        self.analyze_directory(transformers_directory, identifier=files)

    def test_configuration_examples(self):
        if False:
            for i in range(10):
                print('nop')
        transformers_directory = Path('src/transformers')
        files = 'configuration'
        self.analyze_directory(transformers_directory, identifier=files)

    def test_remaining_examples(self):
        if False:
            print('Hello World!')
        transformers_directory = Path('src/transformers')
        n_identifiers = ['configuration', 'modeling', 'tokenization']
        self.analyze_directory(transformers_directory, n_identifier=n_identifiers)

    def test_doc_sources(self):
        if False:
            while True:
                i = 10
        doc_source_directory = Path('docs/source')
        ignore_files = ['favicon.ico']
        self.analyze_directory(doc_source_directory, ignore_files=ignore_files, only_modules=False)