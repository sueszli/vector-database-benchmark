import difflib
from os.path import relpath, join
from pyprint.ConsolePrinter import ConsolePrinter
from coalib.results.Diff import ConflictError
from coalib.results.Result import Result
from coalib.results.result_actions.ResultAction import ResultAction
from coala_utils.decorators import enforce_signature

def format_line(line, real_nr='', sign=']', mod_nr='', symbol=''):
    if False:
        i = 10
        return i + 15
    return '[{:>4}{}{:1}{}'.format(real_nr, sign, symbol, line.rstrip('\n'))

def print_from_name(printer, line):
    if False:
        i = 10
        return i + 15
    printer.print(format_line(line, real_nr='----'), color='red')

def print_to_name(printer, line):
    if False:
        return 10
    printer.print(format_line(line, real_nr='++++'), color='green')

def print_beautified_diff(difflines, printer):
    if False:
        while True:
            i = 10
    current_line_added = None
    current_line_subtracted = None
    for line in difflines:
        if line.startswith('@@'):
            values = line[line.find('-'):line.rfind(' ')]
            (subtracted, added) = tuple(values.split(' '))
            current_line_added = int(added.split(',')[0][1:])
            current_line_subtracted = int(subtracted.split(',')[0][1:])
        elif line.startswith('---'):
            print_from_name(printer, line[4:])
        elif line.startswith('+++'):
            print_to_name(printer, line[4:])
        elif line.startswith('+'):
            printer.print(format_line(line[1:], real_nr=current_line_added), color='green')
            current_line_added += 1
        elif line.startswith('-'):
            printer.print(format_line(line[1:], real_nr=current_line_subtracted), color='red')
            current_line_subtracted += 1
        else:
            current_line_subtracted += 1
            current_line_added += 1

class ShowPatchAction(ResultAction):
    SUCCESS_MESSAGE = 'Displayed patch successfully.'

    @staticmethod
    @enforce_signature
    def is_applicable(result: Result, original_file_dict, file_diff_dict, applied_actions=()):
        if False:
            while True:
                i = 10
        if not result.diffs:
            return 'This result has no patch attached.'
        try:
            nonempty_patches = False
            for (filename, diff) in result.diffs.items():
                if diff and (filename not in file_diff_dict or diff + file_diff_dict[filename] != file_diff_dict[filename]):
                    nonempty_patches = True
            if nonempty_patches:
                return True
            return 'The given patches do not change anything anymore.'
        except ConflictError as ce:
            return f'Two or more patches conflict with each other: {str(ce)}'

    def apply(self, result, original_file_dict, file_diff_dict, no_color: bool=False, show_result_on_top: bool=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        (S)how patch\n\n        :param no_color:\n            Whether or not to use colored output.\n        :param show_result_on_top:\n            Set this to True if you want to show the result info on top.\n            (Useful for e.g. coala_ci.)\n        '
        printer = ConsolePrinter(not no_color)
        if show_result_on_top:
            from coalib.output.ConsoleInteraction import print_result
            print_result(printer, None, {}, result, {}, interactive=False)
        for (filename, this_diff) in sorted(result.diffs.items()):
            to_filename = this_diff.rename if this_diff.rename else filename
            to_filename = '/dev/null' if this_diff.delete else to_filename
            original_file = original_file_dict[filename]
            try:
                current_file = file_diff_dict[filename].modified
                new_file = (file_diff_dict[filename] + this_diff).modified
            except KeyError:
                current_file = original_file
                new_file = this_diff.modified
            if tuple(current_file) != tuple(new_file):
                print_beautified_diff(difflib.unified_diff(current_file, new_file, fromfile=filename, tofile=to_filename), printer)
            elif filename != to_filename:
                print_from_name(printer, join('a', relpath(filename)))
                print_to_name(printer, join('b', relpath(to_filename)))
        return file_diff_dict