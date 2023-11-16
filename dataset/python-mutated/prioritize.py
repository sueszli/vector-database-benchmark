from hscommon.util import dedupe, flatten, rem_file_ext
from hscommon.trans import trget, tr
coltr = trget('columns')

class CriterionCategory:
    NAME = 'Undefined'

    def __init__(self, results):
        if False:
            print('Hello World!')
        self.results = results

    def extract_value(self, dupe):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def format_criterion_value(self, value):
        if False:
            while True:
                i = 10
        return value

    def sort_key(self, dupe, crit_value):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def criteria_list(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

class Criterion:

    def __init__(self, category, value):
        if False:
            i = 10
            return i + 15
        self.category = category
        self.value = value
        self.display_value = category.format_criterion_value(value)

    def sort_key(self, dupe):
        if False:
            i = 10
            return i + 15
        return self.category.sort_key(dupe, self.value)

    @property
    def display(self):
        if False:
            while True:
                i = 10
        return f'{self.category.NAME} ({self.display_value})'

class ValueListCategory(CriterionCategory):

    def sort_key(self, dupe, crit_value):
        if False:
            for i in range(10):
                print('nop')
        if self.extract_value(dupe) == crit_value:
            return 0
        else:
            return 1

    def criteria_list(self):
        if False:
            while True:
                i = 10
        dupes = flatten((g[:] for g in self.results.groups))
        values = sorted(dedupe((self.extract_value(d) for d in dupes)))
        return [Criterion(self, value) for value in values]

class KindCategory(ValueListCategory):
    NAME = coltr('Kind')

    def extract_value(self, dupe):
        if False:
            i = 10
            return i + 15
        value = dupe.extension
        if not value:
            value = tr('None')
        return value

class FolderCategory(ValueListCategory):
    NAME = coltr('Folder')

    def extract_value(self, dupe):
        if False:
            while True:
                i = 10
        return dupe.folder_path

    def format_criterion_value(self, value):
        if False:
            while True:
                i = 10
        return str(value)

    def sort_key(self, dupe, crit_value):
        if False:
            i = 10
            return i + 15
        value = self.extract_value(dupe)
        try:
            value.relative_to(crit_value)
        except ValueError:
            return 1
        return 0

class FilenameCategory(CriterionCategory):
    NAME = coltr('Filename')
    ENDS_WITH_NUMBER = 0
    DOESNT_END_WITH_NUMBER = 1
    LONGEST = 2
    SHORTEST = 3

    def format_criterion_value(self, value):
        if False:
            while True:
                i = 10
        return {self.ENDS_WITH_NUMBER: tr('Ends with number'), self.DOESNT_END_WITH_NUMBER: tr("Doesn't end with number"), self.LONGEST: tr('Longest'), self.SHORTEST: tr('Shortest')}[value]

    def extract_value(self, dupe):
        if False:
            while True:
                i = 10
        return rem_file_ext(dupe.name)

    def sort_key(self, dupe, crit_value):
        if False:
            return 10
        value = self.extract_value(dupe)
        if crit_value in {self.ENDS_WITH_NUMBER, self.DOESNT_END_WITH_NUMBER}:
            ends_with_digit = value.strip()[-1:].isdigit()
            if crit_value == self.ENDS_WITH_NUMBER:
                return 0 if ends_with_digit else 1
            else:
                return 1 if ends_with_digit else 0
        else:
            value = len(value)
            if crit_value == self.LONGEST:
                value *= -1
            return value

    def criteria_list(self):
        if False:
            i = 10
            return i + 15
        return [Criterion(self, crit_value) for crit_value in [self.ENDS_WITH_NUMBER, self.DOESNT_END_WITH_NUMBER, self.LONGEST, self.SHORTEST]]

class NumericalCategory(CriterionCategory):
    HIGHEST = 0
    LOWEST = 1

    def format_criterion_value(self, value):
        if False:
            print('Hello World!')
        return tr('Highest') if value == self.HIGHEST else tr('Lowest')

    def invert_numerical_value(self, value):
        if False:
            for i in range(10):
                print('nop')
        return value * -1

    def sort_key(self, dupe, crit_value):
        if False:
            for i in range(10):
                print('nop')
        value = self.extract_value(dupe)
        if crit_value == self.HIGHEST:
            value = self.invert_numerical_value(value)
        return value

    def criteria_list(self):
        if False:
            print('Hello World!')
        return [Criterion(self, self.HIGHEST), Criterion(self, self.LOWEST)]

class SizeCategory(NumericalCategory):
    NAME = coltr('Size')

    def extract_value(self, dupe):
        if False:
            print('Hello World!')
        return dupe.size

class MtimeCategory(NumericalCategory):
    NAME = coltr('Modification')

    def extract_value(self, dupe):
        if False:
            return 10
        return dupe.mtime

    def format_criterion_value(self, value):
        if False:
            for i in range(10):
                print('nop')
        return tr('Newest') if value == self.HIGHEST else tr('Oldest')

def all_categories():
    if False:
        print('Hello World!')
    return [KindCategory, FolderCategory, FilenameCategory, SizeCategory, MtimeCategory]