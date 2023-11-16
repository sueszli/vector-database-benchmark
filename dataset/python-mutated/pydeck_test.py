import json
from unittest import mock
import pandas as pd
import pydeck as pdk
import streamlit as st
import streamlit.elements.deck_gl_json_chart as deck_gl_json_chart
from tests.delta_generator_test_case import DeltaGeneratorTestCase
df1 = pd.DataFrame({'lat': [1, 2, 3, 4], 'lon': [10, 20, 30, 40]})

class PyDeckTest(DeltaGeneratorTestCase):

    def test_basic(self):
        if False:
            while True:
                i = 10
        'Test that pydeck object works.'
        st.pydeck_chart(pdk.Deck(layers=[pdk.Layer('ScatterplotLayer', data=df1)]))
        el = self.get_delta_from_queue().new_element
        actual = json.loads(el.deck_gl_json_chart.json)
        self.assertEqual(actual['layers'][0]['@@type'], 'ScatterplotLayer')
        self.assertEqual(actual['layers'][0]['data'], [{'lat': 1, 'lon': 10}, {'lat': 2, 'lon': 20}, {'lat': 3, 'lon': 30}, {'lat': 4, 'lon': 40}])
        self.assertEqual(el.deck_gl_json_chart.tooltip, '')

    def test_with_tooltip(self):
        if False:
            return 10
        'Test that pydeck object with tooltip works.'
        tooltip = {'html': '<b>Elevation Value:</b> {elevationValue}', 'style': {'color': 'white'}}
        st.pydeck_chart(pdk.Deck(layers=[pdk.Layer('ScatterplotLayer', data=df1)], tooltip=tooltip))
        el = self.get_delta_from_queue().new_element
        actual = json.loads(el.deck_gl_json_chart.tooltip)
        self.assertEqual(actual, tooltip)

    def test_pydeck_with_tooltip_pydeck_0_7_1(self):
        if False:
            print('Hello World!')
        'Test that pydeck object with tooltip created by pydeck v0.7.1 works.'
        tooltip = {'html': '<b>Elevation Value:</b> {elevationValue}', 'style': {'color': 'white'}}
        mock_desk = mock.Mock(spec=['to_json', '_tooltip'], **{'to_json.return_value': json.dumps({'layers': []}), '_tooltip': tooltip})
        st.pydeck_chart(mock_desk)
        el = self.get_delta_from_queue().new_element
        actual = json.loads(el.deck_gl_json_chart.tooltip)
        self.assertEqual(actual, tooltip)

    def test_pydeck_with_tooltip_pydeck_0_8_1(self):
        if False:
            i = 10
            return i + 15
        'Test that pydeck object with tooltip created by pydeck v0.8.1 works.'
        tooltip = {'html': '<b>Elevation Value:</b> {elevationValue}', 'style': {'color': 'white'}}
        mock_desk = mock.Mock(spec=['to_json', 'deck_widget'], **{'to_json.return_value': json.dumps({'layers': []}), 'deck_widget.tooltip': tooltip})
        st.pydeck_chart(mock_desk)
        el = self.get_delta_from_queue().new_element
        actual = json.loads(el.deck_gl_json_chart.tooltip)
        self.assertEqual(actual, tooltip)

    def test_no_args(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that it can be called with no args.'
        st.pydeck_chart()
        el = self.get_delta_from_queue().new_element
        actual = json.loads(el.deck_gl_json_chart.json)
        self.assertEqual(actual, deck_gl_json_chart.EMPTY_MAP)