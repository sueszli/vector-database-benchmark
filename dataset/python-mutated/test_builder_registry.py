"""Tests of the builder registry."""
import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.builder import builder_registry as registry, HTMLParserTreeBuilder, TreeBuilderRegistry
from . import HTML5LIB_PRESENT, LXML_PRESENT
if HTML5LIB_PRESENT:
    from bs4.builder import HTML5TreeBuilder
if LXML_PRESENT:
    from bs4.builder import LXMLTreeBuilderForXML, LXMLTreeBuilder

class TestBuiltInRegistry(object):
    """Test the built-in registry with the default builders registered."""

    def test_combination(self):
        if False:
            i = 10
            return i + 15
        assert registry.lookup('strict', 'html') == HTMLParserTreeBuilder
        if LXML_PRESENT:
            assert registry.lookup('fast', 'html') == LXMLTreeBuilder
            assert registry.lookup('permissive', 'xml') == LXMLTreeBuilderForXML
        if HTML5LIB_PRESENT:
            assert registry.lookup('html5lib', 'html') == HTML5TreeBuilder

    def test_lookup_by_markup_type(self):
        if False:
            while True:
                i = 10
        if LXML_PRESENT:
            assert registry.lookup('html') == LXMLTreeBuilder
            assert registry.lookup('xml') == LXMLTreeBuilderForXML
        else:
            assert registry.lookup('xml') == None
            if HTML5LIB_PRESENT:
                assert registry.lookup('html') == HTML5TreeBuilder
            else:
                assert registry.lookup('html') == HTMLParserTreeBuilder

    def test_named_library(self):
        if False:
            i = 10
            return i + 15
        if LXML_PRESENT:
            assert registry.lookup('lxml', 'xml') == LXMLTreeBuilderForXML
            assert registry.lookup('lxml', 'html') == LXMLTreeBuilder
        if HTML5LIB_PRESENT:
            assert registry.lookup('html5lib') == HTML5TreeBuilder
        assert registry.lookup('html.parser') == HTMLParserTreeBuilder

    def test_beautifulsoup_constructor_does_lookup(self):
        if False:
            i = 10
            return i + 15
        with warnings.catch_warnings(record=True) as w:
            BeautifulSoup('', features='html')
            BeautifulSoup('', features=['html', 'fast'])
            pass
        with pytest.raises(ValueError):
            BeautifulSoup('', features='no-such-feature')

class TestRegistry(object):
    """Test the TreeBuilderRegistry class in general."""

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.registry = TreeBuilderRegistry()

    def builder_for_features(self, *feature_list):
        if False:
            i = 10
            return i + 15
        cls = type('Builder_' + '_'.join(feature_list), (object,), {'features': feature_list})
        self.registry.register(cls)
        return cls

    def test_register_with_no_features(self):
        if False:
            print('Hello World!')
        builder = self.builder_for_features()
        assert self.registry.lookup('foo') is None
        assert self.registry.lookup() == builder

    def test_register_with_features_makes_lookup_succeed(self):
        if False:
            while True:
                i = 10
        builder = self.builder_for_features('foo', 'bar')
        assert self.registry.lookup('foo') is builder
        assert self.registry.lookup('bar') is builder

    def test_lookup_fails_when_no_builder_implements_feature(self):
        if False:
            print('Hello World!')
        builder = self.builder_for_features('foo', 'bar')
        assert self.registry.lookup('baz') is None

    def test_lookup_gets_most_recent_registration_when_no_feature_specified(self):
        if False:
            while True:
                i = 10
        builder1 = self.builder_for_features('foo')
        builder2 = self.builder_for_features('bar')
        assert self.registry.lookup() == builder2

    def test_lookup_fails_when_no_tree_builders_registered(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.registry.lookup() is None

    def test_lookup_gets_most_recent_builder_supporting_all_features(self):
        if False:
            return 10
        has_one = self.builder_for_features('foo')
        has_the_other = self.builder_for_features('bar')
        has_both_early = self.builder_for_features('foo', 'bar', 'baz')
        has_both_late = self.builder_for_features('foo', 'bar', 'quux')
        lacks_one = self.builder_for_features('bar')
        has_the_other = self.builder_for_features('foo')
        assert self.registry.lookup('foo', 'bar') == has_both_late
        assert self.registry.lookup('foo', 'bar', 'baz') == has_both_early

    def test_lookup_fails_when_cannot_reconcile_requested_features(self):
        if False:
            for i in range(10):
                print('nop')
        builder1 = self.builder_for_features('foo', 'bar')
        builder2 = self.builder_for_features('foo', 'baz')
        assert self.registry.lookup('bar', 'baz') is None