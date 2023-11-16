import json
import os
import re
import unittest

class Dashboard:

    def __init__(self, file):
        if False:
            i = 10
            return i + 15
        self.file = file
        (self.uid, self.queries) = self.get_dashboard_uid_and_queries(file)
        self.regexes = set((self.parse_query_to_regex(query) for query in self.queries))

    @staticmethod
    def get_dashboard_uid_and_queries(file):
        if False:
            i = 10
            return i + 15
        queries = []
        with open(file, 'r') as f:
            data = json.load(f)
            uid = data.get('uid')
            for panel in data.get('panels', []):
                for target in panel.get('targets', []):
                    query = target.get('query')
                    queries.append(query)
        return (uid, queries)

    @staticmethod
    def parse_query_to_regex(query):
        if False:
            for i in range(10):
                print('nop')
        select_pattern = '(.*FROM\\s)(.*)(\\sWHERE.*)'
        match = re.match(select_pattern, query)
        if match:
            from_ = match.group(2)
            without_quotes = re.sub('\\"', '', from_)
            without_retention_policy = without_quotes
            if re.match('(\\w+.\\.)(.*)', without_quotes):
                without_retention_policy = re.sub('(\\w+.)(.*)', '\\2', without_quotes)
            replaced_parameters = re.sub('\\$\\{\\w+\\}', '[\\\\w\\\\d]*', without_retention_policy)
            return replaced_parameters

    @staticmethod
    def _get_json_files_from_directory(directory):
        if False:
            return 10
        return [os.path.join(directory, i) for i in os.listdir(directory) if i.endswith('.json')]

    @classmethod
    def get_dashboards_from_directory(cls, directory):
        if False:
            print('Hello World!')
        for file in cls._get_json_files_from_directory(directory):
            yield cls(file)

def guess_dashboard_by_measurement(measurement, directory, additional_query_substrings=None):
    if False:
        while True:
            i = 10
    '\n  Guesses dashboard by measurement name by parsing queries and matching it with measurement.\n  It is done by using regular expressions obtained from queries.\n  Additionally query can be checked for presence of any of the substrings.\n  '
    dashboards = list(Dashboard.get_dashboards_from_directory(directory))
    ret = []
    for dashboard in dashboards:
        for regex in dashboard.regexes:
            if additional_query_substrings and (not any((substring.lower() in query.lower() for substring in additional_query_substrings for query in dashboard.queries))):
                continue
            if regex and re.match(regex, measurement):
                ret.append(dashboard)
    return list(set(ret))

class TestParseQueryToRegex(unittest.TestCase):

    def test_parse_query_to_regex_1(self):
        if False:
            i = 10
            return i + 15
        query = 'SELECT "runtimeMs" FROM "forever"."nexmark_${ID}_${processingType}" WHERE "runner" =~ /^$runner$/ AND $timeFilter GROUP BY "runner"'
        expected = 'nexmark_[\\w\\d]*_[\\w\\d]*'
        result = Dashboard.parse_query_to_regex(query)
        self.assertEqual(expected, result)

    def test_parse_query_to_regex_2(self):
        if False:
            print('Hello World!')
        query = 'SELECT mean("value") FROM "python_bqio_read_10GB_results" WHERE "metric" =~ /runtime/ AND $timeFilter GROUP BY time($__interval), "metric"'
        expected = 'python_bqio_read_10GB_results'
        result = Dashboard.parse_query_to_regex(query)
        self.assertEqual(expected, result)

    def test_parse_query_to_regex_3(self):
        if False:
            for i in range(10):
                print('nop')
        query = 'SELECT mean("value") FROM "${sdk}_${processingType}_cogbk_3" WHERE "metric" =~ /runtime/ AND $timeFilter GROUP BY time($__interval), "metric"'
        expected = '[\\w\\d]*_[\\w\\d]*_cogbk_3'
        result = Dashboard.parse_query_to_regex(query)
        self.assertEqual(expected, result)