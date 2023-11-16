from typing import NamedTuple, Union
from trashcli.fs import PathExists

def has_been_restored(fs):
    if False:
        for i in range(10):
            print('nop')
    return HasBeenRestoredBaseMatcher(fs, HasBeenRestoredExpectations())

def has_not_been_restored(fs):
    if False:
        return 10
    return HasBeenRestoredBaseMatcher(fs, HasNotBeenYetRestoredExpectations())

class ShouldExists(NamedTuple('ShouldExists', [('name', str), ('path', str)])):

    def expectation_as_text(self):
        if False:
            i = 10
            return i + 15
        return 'should exists'

    def should_exists(self):
        if False:
            print('Hello World!')
        return True

    def actual(self, actually_exists):
        if False:
            while True:
                i = 10
        return {True: 'and it does', False: 'but it does not'}[actually_exists]

class ShouldNotExists(NamedTuple('ShouldNotExists', [('name', str), ('path', str)])):

    def expectation_as_text(self):
        if False:
            for i in range(10):
                print('nop')
        return 'should not exists'

    def should_exists(self):
        if False:
            i = 10
            return i + 15
        return False

    def actual(self, actually_exists):
        if False:
            while True:
                i = 10
        return {False: 'and it does not', True: 'but it does'}[actually_exists]

class Satisfaction:

    def __init__(self, expectation, actually_exists):
        if False:
            for i in range(10):
                print('nop')
        self.expectation = expectation
        self.actually_exists = actually_exists

    def expectations_satisfied(self):
        if False:
            for i in range(10):
                print('nop')
        return self.actually_exists == self.expectation.should_exists()

    def actual_description(self):
        if False:
            return 10
        return self.expectation.actual(self.actually_exists)

    def ok_or_fail_text(self):
        if False:
            while True:
                i = 10
        return {True: 'OK', False: 'FAIL'}[self.expectations_satisfied()]

    def kind_of_file(self):
        if False:
            while True:
                i = 10
        return self.expectation.name

    def satisfaction_description(self):
        if False:
            print('Hello World!')
        return "{0} {1} {2} {3}: '{4}'".format(self.ok_or_fail_text(), self.kind_of_file(), self.expectation.expectation_as_text(), self.actual_description(), self.expectation.path)

class HasBeenRestoredExpectations:

    def expectations_for_file(self, a_trashed_file):
        if False:
            i = 10
            return i + 15
        return [ShouldExists('original_location', a_trashed_file.trashed_from), ShouldNotExists('info_file', a_trashed_file.info_file), ShouldNotExists('backup_copy', a_trashed_file.backup_copy)]

class HasNotBeenYetRestoredExpectations:

    def expectations_for_file(self, a_trashed_file):
        if False:
            return 10
        return [ShouldNotExists('original_location', a_trashed_file.trashed_from), ShouldExists('info_file', a_trashed_file.info_file), ShouldExists('backup_copy', a_trashed_file.backup_copy)]
Expectations = Union[HasBeenRestoredExpectations, HasNotBeenYetRestoredExpectations]

class HasBeenRestoredBaseMatcher:

    def __init__(self, fs, expectations_maker):
        if False:
            while True:
                i = 10
        self.fs = fs
        self.expectations_maker = expectations_maker

    def matches(self, a_trashed_file):
        if False:
            return 10
        return len(self._expectations_failed(a_trashed_file)) == 0

    def describe_mismatch(self, a_trashed_file, focus_on=None):
        if False:
            return 10
        expectations_satisfactions = self._expectations_satisfactions(a_trashed_file, focus_on)
        return 'Expected file to be restore but it has not:\n' + ''.join(('  - %s\n' % satisfaction.satisfaction_description() for satisfaction in expectations_satisfactions))

    def describe(self, description):
        if False:
            i = 10
            return i + 15
        return 'The file has been restored'

    def _expectations_failed(self, a_trashed_file):
        if False:
            while True:
                i = 10
        return [satisfaction for satisfaction in self._expectations_satisfactions(a_trashed_file, focus_on=None) if not satisfaction.expectations_satisfied()]

    def _expectations_satisfactions(self, a_trashed_file, focus_on=None):
        if False:
            return 10
        return [Satisfaction(e, self.fs.exists(e.path)) for e in self._expectations_for(a_trashed_file, focus_on)]

    def _expectations_for(self, a_trashed_file, focus_on=None):
        if False:
            while True:
                i = 10
        all_expectations = self.expectations_maker.expectations_for_file(a_trashed_file)
        if focus_on is None:
            return all_expectations
        else:
            return [e for e in all_expectations if e.name == focus_on]