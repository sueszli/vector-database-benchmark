import numpy as np
import pandas._config.config as cf
from pandas import DataFrame, MultiIndex

class TestTableSchemaRepr:

    def test_publishes(self, ip):
        if False:
            i = 10
            return i + 15
        ipython = ip.instance(config=ip.config)
        df = DataFrame({'A': [1, 2]})
        objects = [df['A'], df]
        expected_keys = [{'text/plain', 'application/vnd.dataresource+json'}, {'text/plain', 'text/html', 'application/vnd.dataresource+json'}]
        opt = cf.option_context('display.html.table_schema', True)
        last_obj = None
        for (obj, expected) in zip(objects, expected_keys):
            last_obj = obj
            with opt:
                formatted = ipython.display_formatter.format(obj)
            assert set(formatted[0].keys()) == expected
        with_latex = cf.option_context('styler.render.repr', 'latex')
        with opt, with_latex:
            formatted = ipython.display_formatter.format(last_obj)
        expected = {'text/plain', 'text/html', 'text/latex', 'application/vnd.dataresource+json'}
        assert set(formatted[0].keys()) == expected

    def test_publishes_not_implemented(self, ip):
        if False:
            i = 10
            return i + 15
        midx = MultiIndex.from_product([['A', 'B'], ['a', 'b', 'c']])
        df = DataFrame(np.random.default_rng(2).standard_normal((5, len(midx))), columns=midx)
        opt = cf.option_context('display.html.table_schema', True)
        with opt:
            formatted = ip.instance(config=ip.config).display_formatter.format(df)
        expected = {'text/plain', 'text/html'}
        assert set(formatted[0].keys()) == expected

    def test_config_on(self):
        if False:
            while True:
                i = 10
        df = DataFrame({'A': [1, 2]})
        with cf.option_context('display.html.table_schema', True):
            result = df._repr_data_resource_()
        assert result is not None

    def test_config_default_off(self):
        if False:
            for i in range(10):
                print('nop')
        df = DataFrame({'A': [1, 2]})
        with cf.option_context('display.html.table_schema', False):
            result = df._repr_data_resource_()
        assert result is None

    def test_enable_data_resource_formatter(self, ip):
        if False:
            for i in range(10):
                print('nop')
        formatters = ip.instance(config=ip.config).display_formatter.formatters
        mimetype = 'application/vnd.dataresource+json'
        with cf.option_context('display.html.table_schema', True):
            assert 'application/vnd.dataresource+json' in formatters
            assert formatters[mimetype].enabled
        assert 'application/vnd.dataresource+json' in formatters
        assert not formatters[mimetype].enabled
        with cf.option_context('display.html.table_schema', True):
            assert 'application/vnd.dataresource+json' in formatters
            assert formatters[mimetype].enabled
            ip.instance(config=ip.config).display_formatter.format(cf)