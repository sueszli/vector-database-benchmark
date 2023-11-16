from coalib.bearlib.spacing.SpacingHelper import SpacingHelper
from coalib.bears.LocalBear import LocalBear
from coalib.results.Diff import Diff
from coalib.results.Result import Result

class SpaceConsistencyTestBear(LocalBear):

    def run(self, filename, file, use_spaces: bool, allow_trailing_whitespace: bool=False, tab_width: int=SpacingHelper.DEFAULT_TAB_WIDTH, enforce_newline_at_EOF: bool=True):
        if False:
            i = 10
            return i + 15
        '\n        Checks the space consistency for each line.\n\n        :param use_spaces:                True if spaces are to be used instead\n                                          of tabs\n        :param allow_trailing_whitespace: Whether to allow trailing whitespace\n                                          or not\n        :param tab_width:                 Number of spaces representing one\n                                          tab\n        :param enforce_newline_at_EOF:    Whether to enforce a newline at the\n                                          End Of File\n        '
        spacing_helper = SpacingHelper(tab_width)
        result_texts = []
        for (line_number, line) in enumerate(file, start=1):
            replacement = line
            if enforce_newline_at_EOF:
                if replacement[-1] != '\n':
                    replacement += '\n'
                    result_texts.append('No newline at EOF.')
            if not allow_trailing_whitespace:
                replacement = replacement.rstrip(' \t\n') + '\n'
                if replacement != line.rstrip('\n') + '\n':
                    result_texts.append('Trailing whitespaces.')
            if use_spaces:
                pre_replacement = replacement
                replacement = spacing_helper.replace_tabs_with_spaces(replacement)
                if replacement != pre_replacement:
                    result_texts.append('Tabs used instead of spaces.')
            else:
                pre_replacement = replacement
                replacement = spacing_helper.replace_spaces_with_tabs(replacement)
                if replacement != pre_replacement:
                    result_texts.append('Spaces used instead of tabs.')
            if len(result_texts) > 0:
                diff = Diff(file)
                diff.change_line(line_number, line, replacement)
                inconsistencies = ''.join(('\n- ' + string for string in result_texts))
                yield Result.from_values(self, 'Line contains following spacing inconsistencies:' + inconsistencies, diffs={filename: diff}, file=filename, line=line_number)
                result_texts = []