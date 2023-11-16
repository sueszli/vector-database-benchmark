from coalib.bearlib.languages import Language
from coalib.bearlib.languages.Language import UnknownLanguageError
from coalib.results.result_actions.ResultAction import ResultAction
from coalib.results.Result import Result
from coalib.results.Diff import Diff
from coala_utils.FileUtils import detect_encoding
from os.path import exists
from os.path import isfile
import shutil
import logging
from coala_utils.decorators import enforce_signature

class IgnoreResultAction(ResultAction):
    SUCCESS_MESSAGE = 'An ignore comment was added to your source code.'

    @staticmethod
    @enforce_signature
    def is_applicable(result: Result, original_file_dict, file_diff_dict, applied_actions=()):
        if False:
            for i in range(10):
                print('nop')
        '\n        For being applicable, the result has to point to a number of files\n        that have to exist i.e. have not been previously deleted.\n        Additionally, the action should not have been applied to the current\n        result before.\n        '
        if IgnoreResultAction.__name__ in applied_actions:
            return 'An ignore comment was already added for this result.'
        if len(result.affected_code) == 0:
            return 'The result is not associated with any source code.'
        filenames = set((src.renamed_file(file_diff_dict) for src in result.affected_code))
        if any((exists(filename) for filename in filenames)):
            return True
        return "The result is associated with source code that doesn't seem to exist."

    def apply(self, result, original_file_dict, file_diff_dict, language: str, no_orig: bool=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add (I)gnore comment\n        '
        ignore_comment = self.get_ignore_comment(result.origin, language)
        if not ignore_comment:
            return file_diff_dict
        source_range = next(filter(lambda sr: exists(sr.file), result.affected_code))
        filename = source_range.file
        ignore_diff = Diff(original_file_dict[filename])
        ignore_diff.change_line(source_range.start.line, original_file_dict[filename][source_range.start.line - 1], original_file_dict[filename][source_range.start.line - 1].rstrip() + '  ' + ignore_comment)
        if filename in file_diff_dict:
            ignore_diff = file_diff_dict[filename] + ignore_diff
        elif not no_orig and isfile(filename):
            shutil.copy2(filename, filename + '.orig')
        file_diff_dict[filename] = ignore_diff
        new_filename = ignore_diff.rename if ignore_diff.rename else filename
        with open(new_filename, mode='w', encoding=detect_encoding(new_filename)) as file:
            file.writelines(ignore_diff.modified)
        return file_diff_dict

    def get_ignore_comment(self, origin, language):
        if False:
            print('Hello World!')
        '\n        Returns a string of Ignore Comment, depending on the language\n        Supports Single Line Comments\n\n        >>> IgnoreResultAction().get_ignore_comment("Bear", "css")\n        \'/* Ignore Bear */\\n\'\n\n        And Multiline Comments\n\n        >>> IgnoreResultAction().get_ignore_comment("Bear", "c")\n        \'// Ignore Bear\\n\'\n        '
        try:
            comment_delimiter = Language[language].get_default_version().comment_delimiters[0]
            ignore_comment = str(comment_delimiter) + ' Ignore ' + origin + '\n'
        except AttributeError:
            try:
                multiline_comment_delimiter = Language[language].get_default_version().multiline_comment_delimiters
                (start_comment, end_comment) = next(iter(multiline_comment_delimiter.items()))
                ignore_comment = str(start_comment) + ' Ignore ' + origin + ' ' + str(end_comment) + '\n'
            except UnknownLanguageError:
                logging.warning(f'coala does not support Ignore in "{language}". Consider opening an issue at https://github.com/coala/coala/issues so we can add support for this language.')
                ignore_comment = None
        return ignore_comment