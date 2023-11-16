from unittest import mock
import plotly.express as px
from parameterized import parameterized
import streamlit as st
from streamlit.errors import StreamlitAPIException
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class PyDeckTest(DeltaGeneratorTestCase):

    def test_basic(self):
        if False:
            while True:
                i = 10
        'Test that plotly object works.'
        df = px.data.gapminder().query("country=='Canada'")
        fig = px.line(df, x='year', y='lifeExp', title='Life expectancy in Canada')
        st.plotly_chart(fig)
        el = self.get_delta_from_queue().new_element
        self.assertNotEqual(el.plotly_chart.figure.spec, None)
        self.assertNotEqual(el.plotly_chart.figure.config, None)

    @parameterized.expand([('streamlit', 'streamlit'), (None, '')])
    def test_theme(self, theme_value, proto_value):
        if False:
            for i in range(10):
                print('nop')
        df = px.data.gapminder().query("country=='Canada'")
        fig = px.line(df, x='year', y='lifeExp', title='Life expectancy in Canada')
        st.plotly_chart(fig, theme=theme_value)
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.plotly_chart.theme, proto_value)

    def test_bad_theme(self):
        if False:
            print('Hello World!')
        df = px.data.gapminder().query("country=='Canada'")
        fig = px.line(df, x='year', y='lifeExp', title='Life expectancy in Canada')
        with self.assertRaises(StreamlitAPIException) as exc:
            st.plotly_chart(fig, theme='bad_theme')
        self.assertEqual(f'You set theme="bad_theme" while Streamlit charts only support theme=”streamlit” or theme=None to fallback to the default library theme.', str(exc.exception))

    def test_st_plotly_chart_simple(self):
        if False:
            print('Hello World!')
        'Test st.plotly_chart.'
        import plotly.graph_objs as go
        trace0 = go.Scatter(x=[1, 2, 3, 4], y=[10, 15, 13, 17])
        data = [trace0]
        st.plotly_chart(data)
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.plotly_chart.HasField('url'), False)
        self.assertNotEqual(el.plotly_chart.figure.spec, '')
        self.assertNotEqual(el.plotly_chart.figure.config, '')
        self.assertEqual(el.plotly_chart.use_container_width, False)

    def test_st_plotly_chart_use_container_width_true(self):
        if False:
            return 10
        'Test st.plotly_chart.'
        import plotly.graph_objs as go
        trace0 = go.Scatter(x=[1, 2, 3, 4], y=[10, 15, 13, 17])
        data = [trace0]
        st.plotly_chart(data, use_container_width=True)
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.plotly_chart.HasField('url'), False)
        self.assertNotEqual(el.plotly_chart.figure.spec, '')
        self.assertNotEqual(el.plotly_chart.figure.config, '')
        self.assertEqual(el.plotly_chart.use_container_width, True)

    def test_st_plotly_chart_sharing(self):
        if False:
            print('Hello World!')
        "Test st.plotly_chart when sending data to Plotly's service."
        import plotly.graph_objs as go
        trace0 = go.Scatter(x=[1, 2, 3, 4], y=[10, 15, 13, 17])
        data = [trace0]
        with mock.patch('streamlit.elements.plotly_chart._plot_to_url_or_load_cached_url') as plot_patch:
            plot_patch.return_value = 'the_url'
            st.plotly_chart(data, sharing='public')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.plotly_chart.HasField('figure'), False)
        self.assertNotEqual(el.plotly_chart.url, 'the_url')
        self.assertEqual(el.plotly_chart.use_container_width, False)