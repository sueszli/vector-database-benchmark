from __future__ import unicode_literals
from __future__ import print_function
from prompt_toolkit.completion import Completer
from .completions import SUBCOMMANDS, ARGS_OPTS_LOOKUP

class Completer(Completer):
    """Completer for haxor-news.

    :type text_utils: :class:`utils.TextUtils`
    :param text_utils: An instance of `utils.TextUtils`.

    :type fuzzy_match: bool
    :param fuzzy_match: Determines whether to use fuzzy matching.
    """

    def __init__(self, fuzzy_match, text_utils):
        if False:
            for i in range(10):
                print('nop')
        self.fuzzy_match = fuzzy_match
        self.text_utils = text_utils

    def completing_command(self, words, word_before_cursor):
        if False:
            for i in range(10):
                print('nop')
        'Determine if we are currently completing the hn command.\n\n        :type words: list\n        :param words: The input text broken into word tokens.\n\n        :type word_before_cursor: str\n        :param word_before_cursor: The current word before the cursor,\n            which might be one or more blank spaces.\n\n        :rtype: bool\n        :return: Specifies whether we are currently completing the hn command.\n        '
        if len(words) == 1 and word_before_cursor != '':
            return True
        else:
            return False

    def completing_subcommand(self, words, word_before_cursor):
        if False:
            return 10
        'Determine if we are currently completing a subcommand.\n\n        :type words: list\n        :param words: The input text broken into word tokens.\n\n        :type word_before_cursor: str\n        :param word_before_cursor: The current word before the cursor,\n            which might be one or more blank spaces.\n\n        :rtype: bool\n        :return: Specifies whether we are currently completing a subcommand.\n        '
        if len(words) == 1 and word_before_cursor == '' or (len(words) == 2 and word_before_cursor != ''):
            return True
        else:
            return False

    def completing_arg(self, words, word_before_cursor):
        if False:
            while True:
                i = 10
        'Determine if we are currently completing an arg.\n\n        :type words: list\n        :param words: The input text broken into word tokens.\n\n        :type word_before_cursor: str\n        :param word_before_cursor: The current word before the cursor,\n            which might be one or more blank spaces.\n\n        :rtype: bool\n        :return: Specifies whether we are currently completing an arg.\n        '
        if len(words) == 2 and word_before_cursor == '' or (len(words) == 3 and word_before_cursor != ''):
            return True
        else:
            return False

    def completing_subcommand_option(self, words, word_before_cursor):
        if False:
            while True:
                i = 10
        'Determine if we are currently completing an option.\n\n        :type words: list\n        :param words: The input text broken into word tokens.\n\n        :type word_before_cursor: str\n        :param word_before_cursor: The current word before the cursor,\n            which might be one or more blank spaces.\n\n        :rtype: list\n        :return: A list of options.\n        '
        options = []
        for (subcommand, args_opts) in ARGS_OPTS_LOOKUP.items():
            if subcommand in words and (words[-2] == subcommand or self.completing_subcommand_option_util(subcommand, words)):
                options.extend(ARGS_OPTS_LOOKUP[subcommand]['opts'])
        return options

    def completing_subcommand_option_util(self, option, words):
        if False:
            for i in range(10):
                print('nop')
        'Determine if we are currently completing an option.\n\n        Called by completing_subcommand_option as a utility method.\n\n        :type option: str\n        :param option: The subcommand in the elements of ARGS_OPTS_LOOKUP.\n\n        :type words: list\n        :param words: The input text broken into word tokens.\n\n        :rtype: bool\n        :return: Specifies whether we are currently completing an option.\n        '
        if len(words) > 3:
            if option in words:
                return True
        return False

    def arg_completions(self, words, word_before_cursor):
        if False:
            i = 10
            return i + 15
        'Generates arguments completions based on the input.\n\n        :type words: list\n        :param words: The input text broken into word tokens.\n\n        :type word_before_cursor: str\n        :param word_before_cursor: The current word before the cursor,\n            which might be one or more blank spaces.\n\n        :rtype: list\n        :return: A list of completions.\n        '
        if 'hn' not in words:
            return []
        for (subcommand, args_opts) in ARGS_OPTS_LOOKUP.items():
            if subcommand in words:
                return [ARGS_OPTS_LOOKUP[subcommand]['args']]
        return ['10']

    def get_completions(self, document, _):
        if False:
            i = 10
            return i + 15
        'Get completions for the current scope.\n\n        :type document: :class:`prompt_toolkit.Document`\n        :param document: An instance of `prompt_toolkit.Document`.\n\n        :type _: :class:`prompt_toolkit.completion.Completion`\n        :param _: (Unused).\n\n        :rtype: generator\n        :return: Yields an instance of `prompt_toolkit.completion.Completion`.\n        '
        word_before_cursor = document.get_word_before_cursor(WORD=True)
        words = self.text_utils.get_tokens(document.text)
        commands = []
        if len(words) == 0:
            return commands
        if self.completing_command(words, word_before_cursor):
            commands = ['hn']
        else:
            if 'hn' not in words:
                return commands
            if self.completing_subcommand(words, word_before_cursor):
                commands = list(SUBCOMMANDS.keys())
            elif self.completing_arg(words, word_before_cursor):
                commands = self.arg_completions(words, word_before_cursor)
            else:
                commands = self.completing_subcommand_option(words, word_before_cursor)
        completions = self.text_utils.find_matches(word_before_cursor, commands, fuzzy=self.fuzzy_match)
        return completions