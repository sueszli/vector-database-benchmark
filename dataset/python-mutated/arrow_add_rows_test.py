"""Unit test of dg.add_rows()."""
import pandas as pd
from parameterized import parameterized
import streamlit as st
from streamlit.type_util import bytes_to_data_frame
from tests.delta_generator_test_case import DeltaGeneratorTestCase
DATAFRAME = pd.DataFrame({'a': [10], 'b': [20], 'c': [30]})
NEW_ROWS = pd.DataFrame({'a': [11, 12, 13], 'b': [21, 22, 23], 'c': [31, 32, 33]})
DATAFRAME2 = pd.DataFrame({'a': [10], 'b': [20], 'c': [30], 'd': [40]})
NEW_ROWS2 = pd.DataFrame({'a': [11, 12, 13], 'b': [21, 22, 23], 'c': [31, 32, 33], 'd': [41, 42, 43]})
ST_CHART_ARGS = [st.area_chart, st.bar_chart, st.line_chart, st.scatter_chart]

class DeltaGeneratorAddRowsTest(DeltaGeneratorTestCase):
    """Test dg.add_rows."""

    @parameterized.expand(ST_CHART_ARGS)
    def test_charts_with_implict_x_and_y(self, chart_command):
        if False:
            return 10
        expected = pd.DataFrame({'index--p5bJXXpQgvPz6yvQMFiy': [1, 2, 3, 1, 2, 3, 1, 2, 3], 'color--p5bJXXpQgvPz6yvQMFiy': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'], 'value--p5bJXXpQgvPz6yvQMFiy': [11, 12, 13, 21, 22, 23, 31, 32, 33]})
        element = chart_command(DATAFRAME)
        element.add_rows(NEW_ROWS)
        proto = bytes_to_data_frame(self.get_delta_from_queue().arrow_add_rows.data.data)
        pd.testing.assert_frame_equal(proto, expected)

    @parameterized.expand(ST_CHART_ARGS)
    def test_charts_with_explicit_x_and_y(self, chart_command):
        if False:
            print('Hello World!')
        expected = pd.DataFrame({'b': [21, 22, 23], 'c': [31, 32, 33]})
        expected.index = pd.RangeIndex(1, 4)
        element = chart_command(DATAFRAME, x='b', y='c')
        element.add_rows(NEW_ROWS)
        proto = bytes_to_data_frame(self.get_delta_from_queue().arrow_add_rows.data.data)
        pd.testing.assert_frame_equal(proto, expected)

    @parameterized.expand(ST_CHART_ARGS)
    def test_charts_with_implict_x_and_explicit_y(self, chart_command):
        if False:
            return 10
        expected = pd.DataFrame({'index--p5bJXXpQgvPz6yvQMFiy': [1, 2, 3], 'b': [21, 22, 23]})
        element = chart_command(DATAFRAME, y='b')
        element.add_rows(NEW_ROWS)
        proto = bytes_to_data_frame(self.get_delta_from_queue().arrow_add_rows.data.data)
        pd.testing.assert_frame_equal(proto, expected)

    @parameterized.expand(ST_CHART_ARGS)
    def test_charts_with_explicit_x_and_implicit_y(self, chart_command):
        if False:
            i = 10
            return i + 15
        expected = pd.DataFrame({'b': [21, 22, 23, 21, 22, 23], 'color--p5bJXXpQgvPz6yvQMFiy': ['a', 'a', 'a', 'c', 'c', 'c'], 'value--p5bJXXpQgvPz6yvQMFiy': [11, 12, 13, 31, 32, 33]})
        element = chart_command(DATAFRAME, x='b')
        element.add_rows(NEW_ROWS)
        proto = bytes_to_data_frame(self.get_delta_from_queue().arrow_add_rows.data.data)
        pd.testing.assert_frame_equal(proto, expected)

    @parameterized.expand(ST_CHART_ARGS)
    def test_charts_with_explicit_x_and_y_sequence(self, chart_command):
        if False:
            print('Hello World!')
        expected = pd.DataFrame({'b': [21, 22, 23, 21, 22, 23], 'color--p5bJXXpQgvPz6yvQMFiy': ['a', 'a', 'a', 'c', 'c', 'c'], 'value--p5bJXXpQgvPz6yvQMFiy': [11, 12, 13, 31, 32, 33]})
        element = chart_command(DATAFRAME, x='b', y=['a', 'c'])
        element.add_rows(NEW_ROWS)
        proto = bytes_to_data_frame(self.get_delta_from_queue().arrow_add_rows.data.data)
        pd.testing.assert_frame_equal(proto, expected)

    @parameterized.expand(ST_CHART_ARGS)
    def test_charts_with_explicit_x_and_y_sequence_and_static_color(self, chart_command):
        if False:
            for i in range(10):
                print('nop')
        expected = pd.DataFrame({'b': [21, 22, 23, 21, 22, 23], 'color--p5bJXXpQgvPz6yvQMFiy': ['a', 'a', 'a', 'c', 'c', 'c'], 'value--p5bJXXpQgvPz6yvQMFiy': [11, 12, 13, 31, 32, 33]})
        element = chart_command(DATAFRAME, x='b', y=['a', 'c'], color=['#f00', '#0f0'])
        element.add_rows(NEW_ROWS)
        proto = bytes_to_data_frame(self.get_delta_from_queue().arrow_add_rows.data.data)
        pd.testing.assert_frame_equal(proto, expected)

    def test_charts_with_explicit_x_and_y_sequence_and_size_set(self):
        if False:
            print('Hello World!')
        expected = pd.DataFrame({'b': [21, 22, 23, 21, 22, 23], 'd': [41, 42, 43, 41, 42, 43], 'color--p5bJXXpQgvPz6yvQMFiy': ['a', 'a', 'a', 'c', 'c', 'c'], 'value--p5bJXXpQgvPz6yvQMFiy': [11, 12, 13, 31, 32, 33]})
        element = st.scatter_chart(DATAFRAME2, x='b', y=['a', 'c'], size='d')
        element.add_rows(NEW_ROWS2)
        proto = bytes_to_data_frame(self.get_delta_from_queue().arrow_add_rows.data.data)
        pd.testing.assert_frame_equal(proto, expected)

    @parameterized.expand(ST_CHART_ARGS)
    def test_charts_with_fewer_args_than_cols(self, chart_command):
        if False:
            for i in range(10):
                print('nop')
        expected = pd.DataFrame({'b': [21, 22, 23], 'a': [11, 12, 13]})
        expected.index = pd.RangeIndex(start=1, stop=4, step=1)
        element = chart_command(DATAFRAME, x='b', y='a')
        element.add_rows(NEW_ROWS)
        proto = bytes_to_data_frame(self.get_delta_from_queue().arrow_add_rows.data.data)
        pd.testing.assert_frame_equal(proto, expected)