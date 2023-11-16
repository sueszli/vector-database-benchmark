"""
Provides a formatter that simplifies to rerun the failing scenarios
of the last test run. It writes a text file with the file locations of
the failing scenarios, like:

    # -- file:rerun.features
    # RERUN: Failing scenarios during last test run.
    features/alice.feature:10
    features/alice.feature:42
    features/bob.feature:67

To rerun the failing scenarios, use:

    behave @rerun_failing.features

Normally, you put the RerunFormatter into the behave configuration file:

    # -- file:behave.ini
    [behave]
    format   = rerun
    outfiles = rerun_failing.features
"""
from __future__ import absolute_import
from datetime import datetime
from os.path import relpath
import os
from behave.formatter.base import Formatter
from behave.model_core import Status

class RerunFormatter(Formatter):
    """
    Provides formatter class that emits a summary which scenarios failed
    during the last test run. This output can be used to rerun the tests
    with the failed scenarios.
    """
    name = 'rerun'
    description = 'Emits scenario file locations of failing scenarios'
    show_timestamp = False
    show_failed_scenarios_descriptions = False

    def __init__(self, stream_opener, config):
        if False:
            print('Hello World!')
        super(RerunFormatter, self).__init__(stream_opener, config)
        self.failed_scenarios = []
        self.current_feature = None

    def reset(self):
        if False:
            while True:
                i = 10
        self.failed_scenarios = []
        self.current_feature = None

    def feature(self, feature):
        if False:
            print('Hello World!')
        self.current_feature = feature

    def eof(self):
        if False:
            while True:
                i = 10
        'Called at end of a feature.'
        if self.current_feature and self.current_feature.status == Status.failed:
            for scenario in self.current_feature.walk_scenarios():
                if scenario.status == Status.failed:
                    self.failed_scenarios.append(scenario)
        self.current_feature = None
        assert self.current_feature is None

    def close(self):
        if False:
            while True:
                i = 10
        'Called at end of test run.'
        stream_name = self.stream_opener.name
        if self.failed_scenarios:
            self.stream = self.open()
            self.report_scenario_failures()
        elif stream_name and os.path.exists(stream_name):
            os.remove(self.stream_opener.name)
        self.close_stream()

    def report_scenario_failures(self):
        if False:
            while True:
                i = 10
        assert self.failed_scenarios
        message = u'# -- RERUN: %d failing scenarios during last test run.\n'
        self.stream.write(message % len(self.failed_scenarios))
        if self.show_timestamp:
            now = datetime.now().replace(microsecond=0)
            self.stream.write('# NOW: %s\n' % now.isoformat(' '))
        if self.show_failed_scenarios_descriptions:
            current_feature = None
            for (index, scenario) in enumerate(self.failed_scenarios):
                if current_feature != scenario.filename:
                    if current_feature is not None:
                        self.stream.write(u'#\n')
                    current_feature = scenario.filename
                    short_filename = relpath(scenario.filename, os.getcwd())
                    self.stream.write(u'# %s\n' % short_filename)
                self.stream.write(u'#  %4d:  %s\n' % (scenario.line, scenario.name))
            self.stream.write('\n')
        for scenario in self.failed_scenarios:
            self.stream.write(u'%s\n' % scenario.location)
        self.stream.write('\n')