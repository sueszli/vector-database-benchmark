from nose.tools import assert_equals, assert_true
from unittest import TestCase
from textblob.en.inflect import plural_categories, singular_ie, singular_irregular, singular_uncountable, singular_uninflected, singularize, pluralize

class InflectTestCase(TestCase):

    def s_singular_pluralize_test(self):
        if False:
            while True:
                i = 10
        assert_equals(pluralize('lens'), 'lenses')

    def s_singular_singularize_test(self):
        if False:
            while True:
                i = 10
        assert_equals(singularize('lenses'), 'lens')

    def diagnoses_singularize_test(self):
        if False:
            i = 10
            return i + 15
        assert_equals(singularize('diagnoses'), 'diagnosis')

    def bus_pluralize_test(self):
        if False:
            while True:
                i = 10
        assert_equals(pluralize('bus'), 'buses')

    def test_all_singular_s(self):
        if False:
            while True:
                i = 10
        for w in plural_categories['s-singular']:
            assert_equals(singularize(pluralize(w)), w)

    def test_all_singular_ie(self):
        if False:
            for i in range(10):
                print('nop')
        for w in singular_ie:
            assert_true(pluralize(w).endswith('ies'))
            assert_equals(singularize(pluralize(w)), w)

    def test_all_singular_irregular(self):
        if False:
            i = 10
            return i + 15
        for singular_w in singular_irregular.values():
            assert_equals(singular_irregular[pluralize(singular_w)], singular_w)