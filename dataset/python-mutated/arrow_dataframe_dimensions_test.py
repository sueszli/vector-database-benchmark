"""Arrow Dataframe dimension parameters test."""
import pandas as pd
import streamlit as st
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class ArrowDataFrameDimensionsTest(DeltaGeneratorTestCase):
    """Test the metadata in the serialized delta message for the different
    dimension specifier options.
    """

    def test_no_dimensions(self):
        if False:
            for i in range(10):
                print('nop')
        'When no dimension parameters are passed'
        self._do_test(lambda fn, df: fn(df), 0, 0)

    def test_with_dimensions(self):
        if False:
            while True:
                i = 10
        'When dimension parameter are passed'
        self._do_test(lambda fn, df: fn(df, 10, 20), 10, 20)

    def test_with_height_only(self):
        if False:
            print('Hello World!')
        'When only height parameter is passed'
        self._do_test(lambda fn, df: fn(df, height=20), 0, 20)

    def test_with_width_only(self):
        if False:
            return 10
        'When only width parameter is passed'
        self._do_test(lambda fn, df: fn(df, width=20), 20, 0)

    def _do_test(self, fn, expectedWidth, expectedHeight):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        fn(st.dataframe, df)
        arrow_data_frame = self.get_delta_from_queue().new_element.arrow_data_frame
        self.assertEqual(arrow_data_frame.width, expectedWidth)
        self.assertEqual(arrow_data_frame.height, expectedHeight)

    def _get_metadata(self):
        if False:
            i = 10
            return i + 15
        'Returns the metadata for the most recent element in the\n        DeltaGenerator queue\n        '
        return self.forward_msg_queue._queue[-1].metadata