"""
Read behave's JSON output files and store retrieved information in
:mod:`behave.model` elements.

Utility to retrieve runtime information from behave's JSON output.

REQUIRES: Python >= 2.6 (json module is part of Python standard library)
"""
from __future__ import absolute_import
import codecs
from behave import model
from behave.model_core import Status
try:
    import json
except ImportError:
    import simplejson as json
__author__ = 'Jens Engel'

def parse(json_filename, encoding='UTF-8'):
    if False:
        while True:
            i = 10
    '\n    Reads behave JSON output file back in and stores information in\n    behave model elements.\n\n    :param json_filename:  JSON filename to process.\n    :return: List of feature objects.\n    '
    with codecs.open(json_filename, 'rU', encoding=encoding) as input_file:
        json_data = json.load(input_file, encoding=encoding)
        json_processor = JsonParser()
        features = json_processor.parse_features(json_data)
        return features

class JsonParser(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.current_scenario_outline = None

    def parse_features(self, json_data):
        if False:
            return 10
        assert isinstance(json_data, list)
        features = []
        json_features = json_data
        for json_feature in json_features:
            feature = self.parse_feature(json_feature)
            features.append(feature)
        return features

    def parse_feature(self, json_feature):
        if False:
            for i in range(10):
                print('nop')
        name = json_feature.get('name', u'')
        keyword = json_feature.get('keyword', None)
        tags = json_feature.get('tags', [])
        description = json_feature.get('description', [])
        location = json_feature.get('location', u'')
        (filename, line) = location.split(':')
        feature = model.Feature(filename, line, keyword, name, tags, description)
        json_elements = json_feature.get('elements', [])
        for json_element in json_elements:
            self.add_feature_element(feature, json_element)
        return feature

    def add_feature_element(self, feature, json_element):
        if False:
            i = 10
            return i + 15
        datatype = json_element.get('type', u'')
        category = datatype.lower()
        if category == 'background':
            background = self.parse_background(json_element)
            feature.background = background
        elif category == 'scenario':
            scenario = self.parse_scenario(json_element)
            feature.add_scenario(scenario)
        elif category == 'scenario_outline':
            scenario_outline = self.parse_scenario_outline(json_element)
            feature.add_scenario(scenario_outline)
            self.current_scenario_outline = scenario_outline
        else:
            raise KeyError('Invalid feature-element keyword: %s' % category)

    def parse_background(self, json_element):
        if False:
            while True:
                i = 10
        "\n        self.add_feature_element({\n            'keyword': background.keyword,\n            'location': background.location,\n            'steps': [],\n        })\n        "
        keyword = json_element.get('keyword', u'')
        name = json_element.get('name', u'')
        location = json_element.get('location', u'')
        json_steps = json_element.get('steps', [])
        steps = self.parse_steps(json_steps)
        (filename, line) = location.split(':')
        background = model.Background(filename, line, keyword, name, steps)
        return background

    def parse_scenario(self, json_element):
        if False:
            for i in range(10):
                print('nop')
        "\n        self.add_feature_element({\n            'keyword': scenario.keyword,\n            'name': scenario.name,\n            'tags': scenario.tags,\n            'location': scenario.location,\n            'steps': [],\n        })\n        "
        keyword = json_element.get('keyword', u'')
        name = json_element.get('name', u'')
        description = json_element.get('description', [])
        tags = json_element.get('tags', [])
        location = json_element.get('location', u'')
        json_steps = json_element.get('steps', [])
        steps = self.parse_steps(json_steps)
        (filename, line) = location.split(':')
        scenario = model.Scenario(filename, line, keyword, name, tags, steps)
        scenario.description = description
        return scenario

    def parse_scenario_outline(self, json_element):
        if False:
            i = 10
            return i + 15
        "\n        self.add_feature_element({\n            'keyword': scenario_outline.keyword,\n            'name': scenario_outline.name,\n            'tags': scenario_outline.tags,\n            'location': scenario_outline.location,\n            'steps': [],\n            'examples': [],\n        })\n        "
        keyword = json_element.get('keyword', u'')
        name = json_element.get('name', u'')
        description = json_element.get('description', [])
        tags = json_element.get('tags', [])
        location = json_element.get('location', u'')
        json_steps = json_element.get('steps', [])
        json_examples = json_element.get('examples', [])
        steps = self.parse_steps(json_steps)
        examples = []
        if json_examples:
            examples = self.parse_examples(json_examples)
        (filename, line) = location.split(':')
        scenario_outline = model.ScenarioOutline(filename, line, keyword, name, tags=tags, steps=steps, examples=examples)
        scenario_outline.description = description
        return scenario_outline

    def parse_steps(self, json_steps):
        if False:
            while True:
                i = 10
        steps = []
        for json_step in json_steps:
            step = self.parse_step(json_step)
            steps.append(step)
        return steps

    def parse_step(self, json_element):
        if False:
            while True:
                i = 10
        "\n        s = {\n            'keyword': step.keyword,\n            'step_type': step.step_type,\n            'name': step.name,\n            'location': step.location,\n        }\n\n        if step.text:\n            s['text'] = step.text\n        if step.table:\n            s['table'] = self.make_table(step.table)\n        element = self.current_feature_element\n        element['steps'].append(s)\n        "
        keyword = json_element.get('keyword', u'')
        name = json_element.get('name', u'')
        step_type = json_element.get('step_type', u'')
        location = json_element.get('location', u'')
        text = json_element.get('text', None)
        if isinstance(text, list):
            text = '\n'.join(text)
        table = None
        json_table = json_element.get('table', None)
        if json_table:
            table = self.parse_table(json_table)
        (filename, line) = location.split(':')
        step = model.Step(filename, line, keyword, step_type, name)
        step.text = text
        step.table = table
        json_result = json_element.get('result', None)
        if json_result:
            self.add_step_result(step, json_result)
        return step

    @staticmethod
    def add_step_result(step, json_result):
        if False:
            for i in range(10):
                print('nop')
        "\n        steps = self.current_feature_element['steps']\n        steps[self._step_index]['result'] = {\n            'status': result.status.name,\n            'duration': result.duration,\n        }\n        "
        status_name = json_result.get('status', u'')
        duration = json_result.get('duration', 0)
        error_message = json_result.get('error_message', None)
        if isinstance(error_message, list):
            error_message = '\n'.join(error_message)
        step.status = Status.from_name(status_name)
        step.duration = duration
        step.error_message = error_message

    @staticmethod
    def parse_table(json_table):
        if False:
            for i in range(10):
                print('nop')
        "\n        table_data = {\n            'headings': table.headings,\n            'rows': [ list(row) for row in table.rows ]\n        }\n        return table_data\n        "
        headings = json_table.get('headings', [])
        rows = json_table.get('rows', [])
        table = model.Table(headings, rows=rows)
        return table

    def parse_examples(self, json_element):
        if False:
            i = 10
            return i + 15
        "\n        e = {\n            'keyword': examples.keyword,\n            'name': examples.name,\n            'location': examples.location,\n        }\n\n        if examples.table:\n            e['table'] = self.make_table(examples.table)\n\n        element = self.current_feature_element\n        element['examples'].append(e)\n        "
        keyword = json_element.get('keyword', u'')
        name = json_element.get('name', u'')
        location = json_element.get('location', u'')
        table = None
        json_table = json_element.get('table', None)
        if json_table:
            table = self.parse_table(json_table)
        (filename, line) = location.split(':')
        examples = model.Examples(filename, line, keyword, name, table)
        return examples