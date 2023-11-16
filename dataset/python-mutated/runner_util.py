"""
Contains utility functions and classes for Runners.
"""
from __future__ import absolute_import, print_function
from bisect import bisect
from collections import OrderedDict
import glob
import os.path
import re
import sys
from six import string_types
from behave import parser
from behave.exception import FileNotFoundError, InvalidFileLocationError, InvalidFilenameError
from behave.model_core import FileLocation
from behave.model import Feature, Rule, ScenarioOutline, Scenario
from behave.textutil import ensure_stream_with_encoder

class FileLocationParser(object):
    pattern = re.compile('^\\s*(?P<filename>.*):(?P<line>\\d+)\\s*$', re.UNICODE)

    @classmethod
    def parse(cls, text):
        if False:
            while True:
                i = 10
        match = cls.pattern.match(text)
        if match:
            filename = match.group('filename').strip()
            line = int(match.group('line'))
            return FileLocation(filename, line)
        filename = text.strip()
        return FileLocation(filename)

class FeatureLineDatabase(object):
    """Helper class that supports select-by-location mechanism (FileLocation)
    within a feature file by storing the feature line numbers for each entity.

    RESPONSIBILITY(s):

    * Can use the line number to select the best matching entity(s) in a feature
    * Implements the select-by-location mechanism for each entity in the feature
    """

    def __init__(self, entity=None, line_data=None):
        if False:
            i = 10
            return i + 15
        if entity and (not line_data):
            line_data = self.make_line_data_for(entity)
        self.entity = entity
        self.data = OrderedDict(line_data or [])
        self._line_numbers = None
        self._line_entities = None

    def select_run_item_by_line(self, line):
        if False:
            for i in range(10):
                print('nop')
        'Select one run-items by using the line number.\n\n        * Exact match returns run-time entity:\n          Feature, Rule, ScenarioOutline, Scenario\n        * Any other line in between uses the predecessor entity\n\n        :param line: Line number in Feature file (as int)\n        :return: Selected run-item object.\n        '
        run_item = self.data.get(line, None)
        if run_item is None:
            if self._line_numbers is None:
                self._line_numbers = list(self.data.keys())
                self._line_entities = list(self.data.values())
            pos = bisect(self._line_numbers, line) - 1
            pos = max(pos, 0)
            run_item = self._line_entities[pos]
        return run_item

    def select_scenarios_by_line(self, line):
        if False:
            for i in range(10):
                print('nop')
        'Select one or more scenarios by using the line number.\n\n        * line = 0: Selects all scenarios in the Feature file\n        * Feature / Rule / ScenarioOutline.location.line selects its scenarios\n        * Scenario.location.line selects the Scenario\n        * Any other lines use the predecessor entity (and its scenarios)\n\n        :param line: Line number in Feature file (as int)\n        :return: List of selected scenarios\n        '
        run_item = self.select_run_item_by_line(line)
        scenarios = []
        if isinstance(run_item, Feature):
            scenarios = list(run_item.walk_scenarios())
        elif isinstance(run_item, Rule):
            scenarios = list(run_item.walk_scenarios())
        elif isinstance(run_item, ScenarioOutline):
            scenarios = list(run_item.scenarios)
        elif isinstance(run_item, Scenario):
            scenarios = [run_item]
        return scenarios

    @classmethod
    def make_line_data_for(cls, entity):
        if False:
            for i in range(10):
                print('nop')
        line_data = []
        run_items = []
        if isinstance(entity, Feature):
            line_data.append((0, entity))
            run_items = entity.run_items
        elif isinstance(entity, Rule):
            run_items = entity.run_items
        elif isinstance(entity, ScenarioOutline):
            run_items = entity.scenarios
        line_data.append((entity.location.line, entity))
        for run_item in run_items:
            line_data.extend(cls.make_line_data_for(run_item))
        return sorted(line_data)

    @classmethod
    def make(cls, entity):
        if False:
            i = 10
            return i + 15
        return cls(entity, cls.make_line_data_for(entity))

class FeatureScenarioLocationCollector(object):
    """
    Collects FileLocation objects for a feature.
    This is used to select a subset of scenarios in a feature that should run.

    USE CASE:
        behave feature/foo.feature:10
        behave @selected_features.txt
        behave @rerun_failed_scenarios.txt

    With features configuration files, like:

        # -- file:rerun_failed_scenarios.txt
        feature/foo.feature:10
        feature/foo.feature:25
        feature/bar.feature
        # -- EOF

    """

    def __init__(self, feature=None, location=None, filename=None):
        if False:
            while True:
                i = 10
        if not filename and location:
            filename = location.filename
        self.feature = feature
        self.filename = filename
        self.use_all_scenarios = False
        self.scenario_lines = set()
        self.all_scenarios = set()
        self.selected_scenarios = set()
        if location:
            self.add_location(location)

    def clear(self):
        if False:
            i = 10
            return i + 15
        self.feature = None
        self.filename = None
        self.use_all_scenarios = False
        self.scenario_lines = set()
        self.all_scenarios = set()
        self.selected_scenarios = set()

    def add_location(self, location):
        if False:
            i = 10
            return i + 15
        if not self.filename:
            self.filename = location.filename
        assert self.filename == location.filename, '%s <=> %s' % (self.filename, location.filename)
        if location.line:
            self.scenario_lines.add(location.line)
        else:
            self.use_all_scenarios = True

    @staticmethod
    def select_scenario_line_for(line, scenario_lines):
        if False:
            i = 10
            return i + 15
        '\n        Select scenario line for any given line.\n\n        ALGORITHM: scenario.line <= line < next_scenario.line\n\n        :param line:  A line number in the file (as number).\n        :param scenario_lines: Sorted list of scenario lines.\n        :return: Scenario.line (first line) for the given line.\n        '
        if not scenario_lines:
            return 0
        pos = bisect(scenario_lines, line) - 1
        pos = max(pos, 0)
        return scenario_lines[pos]

    def discover_selected_scenarios(self, strict=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Discovers selected scenarios based on the provided file locations.\n        In addition:\n          * discover all scenarios\n          * auto-correct BAD LINE-NUMBERS\n\n        :param strict:  If true, raises exception if file location is invalid.\n        :return: List of selected scenarios of this feature (as set).\n        :raises InvalidFileLocationError:\n            If file location is no exactly correct and strict is true.\n        '
        assert self.feature
        if not self.all_scenarios:
            self.all_scenarios = self.feature.walk_scenarios()
        existing_lines = [scenario.line for scenario in self.all_scenarios]
        selected_lines = list(self.scenario_lines)
        for line in selected_lines:
            new_line = self.select_scenario_line_for(line, existing_lines)
            if new_line != line:
                self.scenario_lines.remove(line)
                self.scenario_lines.add(new_line)
                if strict:
                    msg = "Scenario location '...:%d' should be: '%s:%d'" % (line, self.filename, new_line)
                    raise InvalidFileLocationError(msg)
        scenario_lines = set(self.scenario_lines)
        selected_scenarios = set()
        for scenario in self.all_scenarios:
            if scenario.line in scenario_lines:
                selected_scenarios.add(scenario)
                scenario_lines.remove(scenario.line)
        assert not scenario_lines
        return selected_scenarios

    def build_feature(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determines which scenarios in the feature are selected and marks the\n        remaining scenarios as skipped. Scenarios with the following tags\n        are excluded from skipped-marking:\n\n          * @setup\n          * @teardown\n\n        If no file locations are stored, the unmodified feature is returned.\n\n        :return: Feature object to use.\n        '
        use_all_scenarios = not self.scenario_lines or self.use_all_scenarios
        if not self.feature or use_all_scenarios:
            return self.feature
        self.all_scenarios = self.feature.walk_scenarios()
        self.selected_scenarios = self.discover_selected_scenarios()
        unselected_scenarios = set(self.all_scenarios) - self.selected_scenarios
        for scenario in unselected_scenarios:
            if 'setup' in scenario.tags or 'teardown' in scenario.tags:
                continue
            scenario.mark_skipped()
        return self.feature

class FeatureScenarioLocationCollector1(FeatureScenarioLocationCollector):

    @staticmethod
    def select_scenario_line_for(line, scenario_lines):
        if False:
            return 10
        '\n        Select scenario line for any given line.\n\n        ALGORITHM: scenario.line <= line < next_scenario.line\n\n        :param line:  A line number in the file (as number).\n        :param scenario_lines: Sorted list of scenario lines.\n        :return: Scenario.line (first line) for the given line.\n        '
        if not scenario_lines:
            return 0
        pos = bisect(scenario_lines, line) - 1
        pos = max(pos, 0)
        return scenario_lines[pos]

    def discover_selected_scenarios(self, strict=False):
        if False:
            i = 10
            return i + 15
        '\n        Discovers selected scenarios based on the provided file locations.\n        In addition:\n          * discover all scenarios\n          * auto-correct BAD LINE-NUMBERS\n\n        :param strict:  If true, raises exception if file location is invalid.\n        :return: List of selected scenarios of this feature (as set).\n        :raises InvalidFileLocationError:\n            If file location is no exactly correct and strict is true.\n        '
        assert self.feature
        if not self.all_scenarios:
            self.all_scenarios = self.feature.walk_scenarios()
        existing_lines = [scenario.line for scenario in self.all_scenarios]
        selected_lines = list(self.scenario_lines)
        for line in selected_lines:
            new_line = self.select_scenario_line_for(line, existing_lines)
            if new_line != line:
                self.scenario_lines.remove(line)
                self.scenario_lines.add(new_line)
                if strict:
                    msg = "Scenario location '...:%d' should be: '%s:%d'" % (line, self.filename, new_line)
                    raise InvalidFileLocationError(msg)
        scenario_lines = set(self.scenario_lines)
        selected_scenarios = set()
        for scenario in self.all_scenarios:
            if scenario.line in scenario_lines:
                selected_scenarios.add(scenario)
                scenario_lines.remove(scenario.line)
        assert not scenario_lines
        return selected_scenarios

class FeatureScenarioLocationCollector2(FeatureScenarioLocationCollector):

    def discover_selected_scenarios(self, strict=False):
        if False:
            i = 10
            return i + 15
        'Discovers selected scenarios based on the provided file locations.\n        In addition:\n          * discover all scenarios\n          * auto-correct BAD LINE-NUMBERS\n\n        :param strict:  If true, raises exception if file location is invalid.\n        :return: List of selected scenarios of this feature (as set).\n        :raises InvalidFileLocationError:\n            If file location is no exactly correct and strict is true.\n        '
        assert self.feature
        if not self.all_scenarios:
            self.all_scenarios = self.feature.walk_scenarios()
        line_database = FeatureLineDatabase.make(self.feature)
        selected_lines = list(self.scenario_lines)
        selected_scenarios = set()
        for line in selected_lines:
            more_scenarios = line_database.select_scenarios_by_line(line)
            selected_scenarios.update(more_scenarios)
        return selected_scenarios

class FeatureListParser(object):
    """
    Read textual file, ala '@features.txt'. This file contains:

      * a feature filename or FileLocation on each line
      * empty lines (skipped)
      * comment lines (skipped)
      * wildcards are expanded to select 0..N filenames or directories

    Relative path names are evaluated relative to the listfile directory.
    A leading '@' (AT) character is removed from the listfile name.
    """

    @staticmethod
    def parse(text, here=None):
        if False:
            return 10
        '\n        Parse contents of a features list file as text.\n\n        :param text: Contents of a features list(file).\n        :param here: Current working directory to use (optional).\n        :return: List of FileLocation objects\n        '
        locations = []
        for line in text.splitlines():
            filename = line.strip()
            if not filename:
                continue
            if filename.startswith('#'):
                continue
            if here and (not os.path.isabs(filename)):
                filename = os.path.join(here, line)
            filename = os.path.normpath(filename)
            if glob.has_magic(filename):
                for filename2 in glob.iglob(filename):
                    location = FileLocationParser.parse(filename2)
                    locations.append(location)
            else:
                location = FileLocationParser.parse(filename)
                locations.append(location)
        return locations

    @classmethod
    def parse_file(cls, filename):
        if False:
            i = 10
            return i + 15
        "\n        Read textual file, ala '@features.txt'.\n\n        :param filename:  Name of feature list file.\n        :return: List of feature file locations.\n        "
        if filename.startswith('@'):
            filename = filename[1:]
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        here = os.path.dirname(filename) or '.'
        with open(filename) as f:
            contents = f.read()
            return cls.parse(contents, here)

class PathManager(object):
    """Context manager to add paths to sys.path (python search path)
    within a scope.
    """

    def __init__(self, paths=None):
        if False:
            while True:
                i = 10
        self.initial_paths = paths or []
        self.paths = None

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self.paths = list(self.initial_paths)
        sys.path = self.paths + sys.path

    def __exit__(self, *crap):
        if False:
            print('Hello World!')
        for path in self.paths:
            sys.path.remove(path)
        self.paths = None

    def add(self, path):
        if False:
            print('Hello World!')
        if self.paths is None:
            self.initial_paths.append(path)
        else:
            sys.path.insert(0, path)
            self.paths.append(path)

def parse_features(feature_files, language=None):
    if False:
        while True:
            i = 10
    '\n    Parse feature files and return list of Feature model objects.\n    Handles:\n\n      * feature file names, ala "alice.feature"\n      * feature file locations, ala: "alice.feature:10"\n\n    :param feature_files: List of feature file names to parse.\n    :param language:      Default language to use.\n    :return: List of feature objects.\n    '
    scenario_collector = FeatureScenarioLocationCollector2()
    features = []
    for location in feature_files:
        if not isinstance(location, FileLocation):
            assert isinstance(location, string_types)
            location = FileLocation(os.path.normpath(location))
        if location.filename == scenario_collector.filename:
            scenario_collector.add_location(location)
            continue
        if scenario_collector.feature:
            current_feature = scenario_collector.build_feature()
            features.append(current_feature)
            scenario_collector.clear()
        assert isinstance(location, FileLocation)
        filename = os.path.abspath(location.filename)
        feature = parser.parse_file(filename, language=language)
        if feature:
            scenario_collector.feature = feature
            scenario_collector.add_location(location)
    if scenario_collector.feature:
        current_feature = scenario_collector.build_feature()
        features.append(current_feature)
    return features

def collect_feature_locations(paths, strict=True):
    if False:
        while True:
            i = 10
    '\n    Collect feature file names by processing list of paths (from command line).\n    A path can be a:\n\n      * filename (ending with ".feature")\n      * location, ala "{filename}:{line_number}"\n      * features configuration filename, ala "@features.txt"\n      * directory, to discover and collect all "*.feature" files below.\n\n    :param paths:  Paths to process.\n    :return: Feature file locations to use (as list of FileLocations).\n    '
    locations = []
    for path in paths:
        if os.path.isdir(path):
            for (dirpath, dirnames, filenames) in os.walk(path, followlinks=True):
                dirnames.sort()
                for filename in sorted(filenames):
                    if filename.endswith('.feature'):
                        location = FileLocation(os.path.join(dirpath, filename))
                        locations.append(location)
        elif path.startswith('@'):
            locations.extend(FeatureListParser.parse_file(path[1:]))
        else:
            location = FileLocationParser.parse(path)
            if not location.filename.endswith('.feature'):
                raise InvalidFilenameError(location.filename)
            if location.exists():
                locations.append(location)
            elif strict:
                raise FileNotFoundError(path)
    return locations

def exec_file(filename, globals_=None, locals_=None):
    if False:
        print('Hello World!')
    if globals_ is None:
        globals_ = {}
    if locals_ is None:
        locals_ = globals_
    locals_['__file__'] = filename
    with open(filename, 'rb') as f:
        try:
            filename2 = os.path.relpath(filename, os.getcwd())
        except ValueError:
            filename2 = filename
        code = compile(f.read(), filename2, 'exec', dont_inherit=True)
        exec(code, globals_, locals_)

def load_step_modules(step_paths):
    if False:
        for i in range(10):
            print('nop')
    'Load step modules with step definitions from step_paths directories.'
    from behave.api.step_matchers import use_step_matcher, use_default_step_matcher
    from behave.api.step_matchers import step_matcher
    from behave.matchers import use_current_step_matcher_as_default
    from behave.step_registry import setup_step_decorators
    step_globals = {'use_step_matcher': use_step_matcher, 'step_matcher': step_matcher}
    setup_step_decorators(step_globals)
    with PathManager(step_paths):
        use_current_step_matcher_as_default()
        for path in step_paths:
            for name in sorted(os.listdir(path)):
                if name.endswith('.py'):
                    step_module_globals = step_globals.copy()
                    exec_file(os.path.join(path, name), step_module_globals)
                use_default_step_matcher()

def make_undefined_step_snippet(step, language=None):
    if False:
        while True:
            i = 10
    'Helper function to create an undefined-step snippet for a step.\n\n    :param step: Step to use (as Step object or string).\n    :param language: i18n language, optionally needed for step text parsing.\n    :return: Undefined-step snippet (as string).\n    '
    if isinstance(step, string_types):
        step_text = step
        steps = parser.parse_steps(step_text, language=language)
        step = steps[0]
        assert step, 'ParseError: %s' % step_text
    prefix = u'u'
    single_quote = "'"
    if single_quote in step.name:
        step.name = step.name.replace(single_quote, "\\'")
    schema = u"@%s(%s'%s')\ndef step_impl(context):\n"
    schema += u"    raise NotImplementedError(%s'STEP: %s %s')\n\n"
    snippet = schema % (step.step_type, prefix, step.name, prefix, step.step_type.title(), step.name)
    return snippet

def make_undefined_step_snippets(undefined_steps, make_snippet=None):
    if False:
        i = 10
        return i + 15
    'Creates a list of undefined step snippets.\n    Note that duplicated steps are removed internally.\n\n    :param undefined_steps: List of undefined steps (as Step object or string).\n    :param make_snippet:    Function that generates snippet (optional)\n    :return: List of undefined step snippets (as list of strings)\n    '
    if make_snippet is None:
        make_snippet = make_undefined_step_snippet
    step_snippets = []
    collected_steps = set()
    for undefined_step in undefined_steps:
        if undefined_step in collected_steps:
            continue
        collected_steps.add(undefined_step)
        step_snippet = make_snippet(undefined_step)
        step_snippets.append(step_snippet)
    return step_snippets

def print_undefined_step_snippets(undefined_steps, stream=None, colored=True):
    if False:
        return 10
    '\n    Print snippets for the undefined steps that were discovered.\n\n    :param undefined_steps:  List of undefined steps (as list<string>).\n    :param stream:      Output stream to use (default: sys.stderr).\n    :param colored:     Indicates if coloring should be used (default: True)\n    '
    if not undefined_steps:
        return
    if not stream:
        stream = sys.stderr
    msg = u'\nYou can implement step definitions for undefined steps with '
    msg += u'these snippets:\n\n'
    msg += u'\n'.join(make_undefined_step_snippets(undefined_steps))
    if colored:
        from behave.formatter.ansi_escapes import escapes
        msg = escapes['undefined'] + msg + escapes['reset']
    stream = ensure_stream_with_encoder(stream)
    stream.write(msg)
    stream.flush()

def reset_runtime():
    if False:
        i = 10
        return i + 15
    'Reset runtime environment.\n    Best effort to reset module data to initial state.\n    '
    from behave import step_registry
    from behave import matchers
    step_registry.registry = step_registry.StepRegistry()
    step_registry.setup_step_decorators(None, step_registry.registry)
    matchers.get_matcher_factory().reset()