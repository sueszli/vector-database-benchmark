from mage_ai.data_cleaner.cleaning_rules.base import BaseRule
from mage_ai.data_cleaner.column_types.constants import ColumnType
from mage_ai.data_cleaner.transformer_actions.constants import ActionType, Axis
import pandas as pd
import re

class ReformatValuesSubRule:
    """
    Assumptions of ReformatValuesSubRule
    1. df will not contain any empty strings - all empty strings are converted to null types.
    This is handled in ImputeValues.
    2. column_types will contain the correct type value
    3. Every column in df is of dtype object and the entries must be used to infer type.
    This is not always the case, but this assumption simplifies code
    """

    def __init__(self, action_builder, clean_column_cache, column_types, df, exact_dtypes, statistics):
        if False:
            return 10
        self.df = df
        self.column_types = column_types
        self.exact_dtypes = exact_dtypes
        self.statistics = statistics
        self.action_builder = action_builder
        self.clean_column_cache = clean_column_cache
        self.matches = []

    def clean_column(self, column):
        if False:
            i = 10
            return i + 15
        '\n        Removes all null entries from a specific column\n        '
        return self.clean_column_cache.setdefault(column, self.df[column].dropna(axis=0))

    def evaluate(self, column):
        if False:
            print('Hello World!')
        raise NotImplementedError('Children of ReformatValuesSubRule must override this method.')

    def get_suggestions(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('Children of ReformatValuesSubRule must override this method.')

class StandardizeCapitalizationSubRule(ReformatValuesSubRule):
    UPPERCASE_PATTERN = '^[^a-z]*$'
    LOWERCASE_PATTERN = '^[^A-Z]*$'
    NON_ALPH_PATTERN = '[^A-Za-z]'
    ALPHABETICAL_TYPES = frozenset((ColumnType.CATEGORY_HIGH_CARDINALITY, ColumnType.CATEGORY, ColumnType.TEXT, ColumnType.EMAIL))
    NON_ALPH_UB = 0.4
    ALPH_RATIO_LB = 0.6

    def __init__(self, action_builder, clean_column_cache, column_types, df, exact_dtypes, statistics):
        if False:
            print('Hello World!')
        super().__init__(action_builder, clean_column_cache, column_types, df, exact_dtypes, statistics)
        self.uppercase = []
        self.lowercase = []

    def filter_column_regex(self, df_column, regex_pattern):
        if False:
            print('Hello World!')
        if df_column.empty:
            return (0, df_column)
        meets_regex = df_column.str.match(regex_pattern)
        try:
            count = meets_regex.value_counts()[True]
        except KeyError:
            count = 0
        return (count, df_column[~meets_regex])

    def evaluate(self, column):
        if False:
            print('Hello World!')
        '\n        Rule:\n        1. If column is not a category/string type which may have alphabet, no suggestion\n        2. If non-null entries are not string, no suggestion\n        3. If NON_ALPH_UB of entries are not majority alphabetical, no suggestion. Majority\n            alphabetical == ALPH_RATIO_LB of all chars are alphabet\n        4. If all entries are same case, no suggestion\n        5. Suggest the more prevalent occurrence (e.g., if most alphabetical entries are lowercase\n        text but some mixedcase and uppercase text, suggest conversion to lowercase)\n        5a. If most alphabetical entries are mixedcase, suggest conversion to lowercase\n        '
        dtype = self.column_types[column]
        if dtype not in self.ALPHABETICAL_TYPES:
            return
        clean_col = self.clean_column(column)
        if self.exact_dtypes[column] is not str:
            return
        non_alpha_ratio = clean_col.str.count(self.NON_ALPH_PATTERN) / clean_col.str.len()
        unfiltered_length = self.statistics[f'{column}/count']
        clean_col = clean_col[non_alpha_ratio <= self.NON_ALPH_UB]
        new_length = clean_col.count()
        if new_length / unfiltered_length <= self.ALPH_RATIO_LB:
            return
        (uppercase, clean_col) = self.filter_column_regex(clean_col, self.UPPERCASE_PATTERN)
        (lowercase, clean_col) = self.filter_column_regex(clean_col, self.LOWERCASE_PATTERN)
        mixedcase = clean_col.count()
        uppercase_ratio = uppercase / new_length
        lowercase_ratio = lowercase / new_length
        mixedcase_ratio = mixedcase / new_length
        if uppercase_ratio != 1 and lowercase_ratio != 1:
            max_case_style = max(uppercase_ratio, lowercase_ratio, mixedcase_ratio)
            if max_case_style == uppercase_ratio:
                self.uppercase.append(column)
            else:
                self.lowercase.append(column)

    def get_suggestions(self):
        if False:
            print('Hello World!')
        suggestions = []
        payloads = {'uppercase': self.uppercase, 'lowercase': self.lowercase}
        for case in payloads:
            if len(payloads[case]) != 0:
                suggestions.append(self.action_builder('Reformat values', f'Format entries in these columns as fully {case} to improve data quality.', ActionType.REFORMAT, action_arguments=payloads[case], axis=Axis.COLUMN, action_options={'reformat': 'caps_standardization', 'capitalization': case}))
        return suggestions

class ConvertCurrencySubRule(ReformatValuesSubRule):
    CURR_PREFIX = '(?:[\\$\\€\\¥\\₹\\£]|(?:Rs)|(?:CAD))'
    CURR_SUFFIX = '(?:[\\元\\€\\$]|(?:CAD))'
    NUMBER_PATTERN = '[0-9]*\\.{0,1}[0-9]+'
    CURRENCY_BODY = f'(?:{CURR_PREFIX}\\s*{NUMBER_PATTERN}|{NUMBER_PATTERN}\\s*{CURR_SUFFIX})'
    CURRENCY_PATTERN = re.compile(f'^\\s*(?:\\-*\\s*{CURRENCY_BODY}|{CURRENCY_BODY}\\s*)\\s*$')
    CURRENCY_TYPES = frozenset((ColumnType.CATEGORY, ColumnType.CATEGORY_HIGH_CARDINALITY, ColumnType.TEXT, ColumnType.NUMBER, ColumnType.NUMBER_WITH_DECIMALS))

    def evaluate(self, column):
        if False:
            for i in range(10):
                print('nop')
        "\n        Rule:\n        1. If the column is not a text, number, or category type, no suggestion\n        2. If the column is not a string type, it can't contain currency symbol, no suggestion\n        3. If all entries are of currency type (currency symbol followed by number),\n           suggest conversion to number_with_decimal; else don't\n        "
        dtype = self.column_types[column]
        if dtype not in self.CURRENCY_TYPES:
            return
        clean_col = self.clean_column(column)
        if self.exact_dtypes[column] is not str:
            return
        currency_pattern_mask = clean_col.str.match(self.CURRENCY_PATTERN)
        try:
            count = currency_pattern_mask.value_counts()[True]
        except KeyError:
            count = 0
        if count / self.statistics[f'{column}/count'] == 1:
            self.matches.append(column)

    def get_suggestions(self):
        if False:
            return 10
        suggestions = []
        if len(self.matches) != 0:
            suggestions.append(self.action_builder('Reformat values', 'Format entries in these columns as numbers to improve data quality.', ActionType.REFORMAT, action_arguments=self.matches, axis=Axis.COLUMN, action_options={'reformat': 'currency_to_num'}))
        return suggestions

class ReformatDateSubRule(ReformatValuesSubRule):
    DATE_MATCHES_LB = 0.3
    DATE_TYPES = frozenset((ColumnType.DATETIME,))

    def evaluate(self, column):
        if False:
            print('Hello World!')
        '\n        Rule:\n        1. If column is not of dtype category or text, no suggestion\n        2. If column does not contain string types, no suggestion\n        3. Try use Pandas datetime parse to convert from string to datetime.\n           If more than DATE_MATCHES_LB entries are succesfully converted, suggest\n           conversion to datetime type\n        '
        dtype = self.column_types[column]
        if dtype not in self.DATE_TYPES:
            return
        if not self.exact_dtypes[column] is str:
            return
        clean_col = self.strip_column_for_date_parsing(column)
        clean_col = pd.to_datetime(clean_col, infer_datetime_format=True, errors='coerce')
        notnull_value_rate = clean_col.count() / len(clean_col)
        if notnull_value_rate >= self.DATE_MATCHES_LB:
            self.matches.append(column)

    def get_suggestions(self):
        if False:
            i = 10
            return i + 15
        suggestions = []
        if len(self.matches) != 0:
            suggestions.append(self.action_builder('Reformat values', 'Format entries in these columns as datetime objects to improve data quality.', ActionType.REFORMAT, action_arguments=self.matches, axis=Axis.COLUMN, action_options={'reformat': 'date_format_conversion'}))
        return suggestions

    def strip_column_for_date_parsing(self, column):
        if False:
            for i in range(10):
                print('nop')
        clean_col = self.clean_column(column)
        clean_col = clean_col.str.replace('[\\,\\s\\t]+', ' ')
        clean_col = clean_col.str.replace('\\s*([\\/\\\\\\-\\.]+)\\s*', lambda group: group.group(1)[0])
        return clean_col.str.lower()

class ReformatValues(BaseRule):
    RULE_LIST = [StandardizeCapitalizationSubRule, ConvertCurrencySubRule, ReformatDateSubRule]

    def __init__(self, df, column_types, statistics, custom_config={}):
        if False:
            print('Hello World!')
        super().__init__(df, column_types, statistics, custom_config=custom_config)
        self.clean_column_cache = {}
        self.exact_dtypes = self.infer_exact_dtypes()

    def hydrate_rule_list(self):
        if False:
            print('Hello World!')
        return list(map(lambda x: x(self._build_transformer_action_suggestion, self.clean_column_cache, self.column_types, self.df, self.exact_dtypes, self.statistics), self.RULE_LIST))

    def evaluate(self):
        if False:
            return 10
        rules = self.hydrate_rule_list()
        for column in self.df_columns:
            for rule in rules:
                rule.evaluate(column)
        suggestions = []
        for rule in rules:
            suggestions.extend(rule.get_suggestions())
        return suggestions

    def infer_exact_dtypes(self):
        if False:
            return 10
        exact_dtypes = {}
        for column in self.df_columns:
            clean_col = self.df[column].dropna(axis=0)
            try:
                dtype = type(clean_col.iloc[0])
            except IndexError:
                dtype = None
            exact_dtypes[column] = dtype
        return exact_dtypes