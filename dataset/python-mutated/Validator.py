from lux.core.frame import LuxDataFrame
from lux.vis.Clause import Clause
from typing import List
from lux.utils.date_utils import is_datetime_series, is_datetime_string
import warnings
import pandas as pd
import lux
import lux.utils.utils

class Validator:
    """
    Contains methods for validating lux.Clause objects in the intent.
    """

    def __init__(self):
        if False:
            return 10
        self.name = 'Validator'
        warnings.formatwarning = lux.warning_format

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'<Validator>'

    @staticmethod
    def validate_intent(intent: List[Clause], ldf: LuxDataFrame, suppress_warning=False):
        if False:
            i = 10
            return i + 15
        '\n        Validates input specifications from the user to find inconsistencies and errors.\n\n        Parameters\n        ----------\n        ldf : lux.core.frame\n                LuxDataFrame with underspecified intent.\n\n        Returns\n        -------\n        Boolean\n                True if the intent passed in is valid, False otherwise.\n\n        Raises\n        ------\n        ValueError\n                Ensures input intent are consistent with DataFrame content.\n\n        '

        def validate_clause(clause):
            if False:
                for i in range(10):
                    print('nop')
            warn_msg = ''
            if not (clause.attribute == '?' or clause.value == '?' or clause.attribute == ''):
                if isinstance(clause.attribute, list):
                    for attr in clause.attribute:
                        if attr not in list(ldf.columns):
                            warn_msg = f"\n- The input attribute '{attr}' does not exist in the DataFrame."
                elif clause.attribute != 'Record':
                    if isinstance(clause.attribute, str) and (not is_datetime_string(clause.attribute)):
                        if not clause.attribute in list(ldf.columns):
                            search_val = clause.attribute
                            match_attr = False
                            for (attr, val_list) in ldf.unique_values.items():
                                if search_val in val_list:
                                    match_attr = attr
                            if match_attr:
                                warn_msg = f"\n- The input '{search_val}' looks like a value that belongs to the '{match_attr}' attribute. \n  Please specify the value fully, as something like {match_attr}={search_val}."
                            else:
                                warn_msg = f"\n- The input attribute '{clause.attribute}' does not exist in the DataFrame. \n  Please check your input intent for typos."
                    if clause.value != '' and clause.attribute != '' and (clause.filter_op == '='):
                        if not lux.utils.utils.like_nan(clause.value):
                            series = ldf[clause.attribute]
                            if not is_datetime_series(series):
                                if isinstance(clause.value, list):
                                    vals = clause.value
                                else:
                                    vals = [clause.value]
                                for val in vals:
                                    if lux.config.executor.name == 'PandasExecutor' and val not in series.values:
                                        warn_msg = f"\n- The input value '{val}' does not exist for the attribute '{clause.attribute}' for the DataFrame."
            return warn_msg
        warn_msg = ''
        for clause in intent:
            if type(clause) is list:
                for s in clause:
                    warn_msg += validate_clause(s)
            else:
                warn_msg += validate_clause(clause)
        if warn_msg != '' and (not suppress_warning):
            warnings.warn('\nThe following issues are ecountered when validating the parsed intent:' + warn_msg, stacklevel=2)
        return warn_msg == ''