from pygments.lexers import guess_lexer_for_filename
from pyprint.ConsolePrinter import ConsolePrinter
from coalib.results.result_actions.ResultAction import ResultAction
from coalib.results.result_actions.ShowPatchAction import ShowPatchAction
from coalib.output.printers.LogPrinter import LogPrinter
from coalib.parsing.FilterHelper import apply_filters
from coalib.coala_modes import mode_normal
from coalib.parsing.DefaultArgParser import default_arg_parser
OBJECT_INDEX = FILENAME_INDEX = 0
DEFAULT_BEAR = 'SpaceConsistencyBear'

def show_possibilities(console_printer, i, action):
    if False:
        print('Hello World!')
    console_printer.print("[{:>4}]  {}. Apply patch ('{}')".format('', i, action))

def create_arg_parser(files, bears):
    if False:
        while True:
            i = 10
    '\n    A function that generates a `default_arg_parser`.\n\n    :param files: A list that contains filenames.\n    :param bears: A list that contains name of bears.\n    :return:      An object of type `default_arg_parser`.\n    '
    args = default_arg_parser().parse_args()
    args.files = files
    args.bears = bears
    args.default_actions = '**: ApplyPatchAction'
    return args

def filter_bears(language):
    if False:
        i = 10
        return i + 15
    '\n    Filter bears by language.\n\n    :param language: The language to filter with.\n    :return:         A list of bears.\n    '
    return list(apply_filters({'language': (language,)}, None)[0]['cli'])

def find_language(filename):
    if False:
        while True:
            i = 10
    '\n    Find the language used in `filename`.\n\n    :param filename: The name of the file.\n    :return:         The language used.\n    '
    return guess_lexer_for_filename(filename, 'Error, no file found').name

class DefaultBear:

    def __init__(self):
        if False:
            return 10
        self.name = DEFAULT_BEAR

class GeneratePatchesAction(ResultAction):
    SUCCESS_MESSAGE = 'Patch generated successfully.'
    is_applicable = staticmethod(ShowPatchAction.is_applicable)

    def apply(self, result, original_file_dict, file_diff_dict):
        if False:
            return 10
        '\n        (G)enerate patches\n        '
        console_printer = ConsolePrinter()
        log_printer = LogPrinter()
        to_filename = sorted(result.diffs.items())[OBJECT_INDEX][FILENAME_INDEX]
        filtered_bears = filter_bears(find_language(to_filename))
        filtered_bears.insert(0, DefaultBear())
        possible_options = [b.name for b in filtered_bears]
        console_printer.print('[{:>4}] *0. Do Nothing'.format(''))
        for (i, action) in enumerate(possible_options, 1):
            show_possibilities(console_printer, i, action)
        choose_action = str(input('[{:>4}]  Enter a number: '.format('')))
        if choose_action is '' or choose_action is '0':
            return False
        choose_action = int(choose_action)
        chosen_bear = [possible_options[choose_action - 1]]
        return mode_normal(console_printer, log_printer, create_arg_parser([to_filename], chosen_bear), debug=False)