import numpy as np
import pandas as pd
from featuretools.primitives import CountString
from featuretools.tests.primitive_tests.utils import PrimitiveTestBase, find_applicable_primitives, valid_dfs

class TestCountString(PrimitiveTestBase):
    primitive = CountString

    def compare(self, primitive_initiated, test_cases, answers):
        if False:
            i = 10
            return i + 15
        primitive_func = primitive_initiated.get_function()
        primitive_answers = primitive_func(test_cases)
        return np.testing.assert_array_equal(answers, primitive_answers)
    test_cases = pd.Series(['Hello other words hello hEllo HELLO', 'he\\{ll\t\n\t.--?o othe/r words hello hello h.el./lo', 'hellohellohello other hello word go hello here 9hello hello9', "helloHellohello 9Hello 9hello9 *hello/ test'hel..lo' 'hE.l.lO'          hello"])

    def test_non_regex_with_no_other_parameters(self):
        if False:
            return 10
        primitive = self.primitive('hello', ignore_case=False, ignore_non_alphanumeric=False, is_regex=False, match_whole_words_only=False)
        answers = [1, 2, 7, 5]
        self.compare(primitive, self.test_cases, answers)

    def test_non_regex_ignore_case(self):
        if False:
            i = 10
            return i + 15
        primitive1 = self.primitive('hello', ignore_case=True, ignore_non_alphanumeric=False, is_regex=False, match_whole_words_only=False)
        primitive2 = self.primitive('HeLLo', ignore_case=True, ignore_non_alphanumeric=False, is_regex=False, match_whole_words_only=False)
        answers = [4, 2, 7, 7]
        self.compare(primitive1, self.test_cases, answers)
        self.compare(primitive2, self.test_cases, answers)

    def test_non_regex_ignore_non_alphanumeric(self):
        if False:
            return 10
        primitive = self.primitive('hello', ignore_case=False, ignore_non_alphanumeric=True, is_regex=False, match_whole_words_only=False)
        answers = [1, 4, 7, 6]
        self.compare(primitive, self.test_cases, answers)

    def test_non_regex_match_whole_words_only(self):
        if False:
            print('Hello World!')
        primitive = self.primitive('hello', ignore_case=False, ignore_non_alphanumeric=False, is_regex=False, match_whole_words_only=True)
        answers = [1, 2, 2, 2]
        self.compare(primitive, self.test_cases, answers)

    def test_non_regex_with_all_others_parameters(self):
        if False:
            print('Hello World!')
        primitive = self.primitive('hello', ignore_case=True, ignore_non_alphanumeric=True, is_regex=False, match_whole_words_only=True)
        answers = [4, 4, 2, 3]
        self.compare(primitive, self.test_cases, answers)

    def test_regex_with_no_other_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        primitive = self.primitive('h.l.o', ignore_case=False, ignore_non_alphanumeric=False, is_regex=True, match_whole_words_only=False)
        answers = [2, 2, 7, 5]
        self.compare(primitive, self.test_cases, answers)

    def test_regex_with_ignore_case(self):
        if False:
            while True:
                i = 10
        primitive = self.primitive('h.l.o', ignore_case=True, ignore_non_alphanumeric=False, is_regex=True, match_whole_words_only=False)
        answers = [4, 2, 7, 7]
        self.compare(primitive, self.test_cases, answers)

    def test_regex_with_ignore_non_alphanumeric(self):
        if False:
            i = 10
            return i + 15
        primitive = self.primitive('h.l.o', ignore_case=False, ignore_non_alphanumeric=True, is_regex=True, match_whole_words_only=False)
        answers = [2, 4, 7, 6]
        self.compare(primitive, self.test_cases, answers)

    def test_regex_with_match_whole_words_only(self):
        if False:
            i = 10
            return i + 15
        primitive = self.primitive('h.l.o', ignore_case=False, ignore_non_alphanumeric=False, is_regex=True, match_whole_words_only=True)
        answers = [2, 2, 2, 2]
        self.compare(primitive, self.test_cases, answers)

    def test_regex_with_all_other_parameters(self):
        if False:
            while True:
                i = 10
        primitive = self.primitive('h.l.o', ignore_case=True, ignore_non_alphanumeric=True, is_regex=True, match_whole_words_only=True)
        answers = [4, 4, 2, 3]
        self.compare(primitive, self.test_cases, answers)

    def test_overlapping_regex(self):
        if False:
            i = 10
            return i + 15
        primitive = self.primitive('(?=(a.*a))', ignore_case=True, ignore_non_alphanumeric=True, is_regex=True, match_whole_words_only=False)
        test_cases = pd.Series(['aaaaaaaaaa', 'atesta aa aa a'])
        answers = [9, 6]
        self.compare(primitive, test_cases, answers)

    def test_the(self):
        if False:
            for i in range(10):
                print('nop')
        primitive = self.primitive('the', ignore_case=True, ignore_non_alphanumeric=False, is_regex=False, match_whole_words_only=False)
        test_cases = pd.Series(['The fox jumped over the cat', 'The there then'])
        answers = [2, 3]
        self.compare(primitive, test_cases, answers)

    def test_nan(self):
        if False:
            for i in range(10):
                print('nop')
        primitive = self.primitive('the', ignore_case=True, ignore_non_alphanumeric=False, is_regex=False, match_whole_words_only=False)
        test_cases = pd.Series([np.nan, None, pd.NA, 'The fox jumped over the cat', 'The there then'])
        answers = [np.nan, np.nan, np.nan, 2, 3]
        self.compare(primitive, test_cases, answers)

    def test_with_featuretools(self, es):
        if False:
            print('Hello World!')
        (transform, aggregation) = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive('the', ignore_case=True, ignore_non_alphanumeric=False, is_regex=False, match_whole_words_only=False)
        transform.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)

    def test_with_featuretools_nan(self, es):
        if False:
            i = 10
            return i + 15
        log_df = es['log']
        comments = log_df['comments']
        comments[1] = pd.NA
        comments[2] = np.nan
        comments[3] = None
        log_df['comments'] = comments
        es.replace_dataframe(dataframe_name='log', df=log_df)
        (transform, aggregation) = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive('the', ignore_case=True, ignore_non_alphanumeric=False, is_regex=False, match_whole_words_only=False)
        transform.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)