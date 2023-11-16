"""st.pyplot unit tests."""
from typing import Optional
from unittest.mock import patch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from parameterized import parameterized
import streamlit as st
from streamlit.elements import image
from streamlit.web.server.server import MEDIA_ENDPOINT
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class PyplotTest(DeltaGeneratorTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        if matplotlib.get_backend().lower() != 'agg':
            plt.switch_backend('agg')

    def tearDown(self):
        if False:
            while True:
                i = 10
        plt.clf()
        super().tearDown()

    def test_st_pyplot(self):
        if False:
            return 10
        'Test st.pyplot.\n\n        Need to test:\n        * Failed import of matplotlib.\n        * Passing in a figure.\n        '
        np.random.seed(19680801)
        data = np.random.randn(2, 20)
        (fig, ax) = plt.subplots(figsize=(2, 2))
        ax.scatter(data[0], data[1])
        st.pyplot(fig)
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.imgs.width, -2)
        self.assertEqual(el.imgs.imgs[0].caption, '')
        self.assertTrue(el.imgs.imgs[0].url.startswith(MEDIA_ENDPOINT))

    @parameterized.expand([('true', True), ('false', False), ('none', None)])
    def test_st_pyplot_clear_global_figure(self, _, clear_figure: Optional[bool]):
        if False:
            print('Hello World!')
        'st.pyplot should clear the global figure if `clear_figure` is\n        True *or* None.\n        '
        plt.hist(np.random.normal(1, 1, size=100), bins=20)
        with patch.object(plt, 'clf', wraps=plt.clf, autospec=True) as plt_clf:
            st.pyplot(clear_figure=clear_figure)
            if clear_figure in (True, None):
                plt_clf.assert_called_once()
            else:
                plt_clf.assert_not_called()

    @parameterized.expand([('true', True), ('false', False), ('none', None)])
    def test_st_pyplot_clear_figure(self, _, clear_figure: Optional[bool]):
        if False:
            for i in range(10):
                print('nop')
        'st.pyplot should clear the passed-in figure if `clear_figure` is True.'
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.hist(np.random.normal(1, 1, size=100), bins=20)
        with patch.object(fig, 'clf', wraps=fig.clf, autospec=True) as fig_clf:
            st.pyplot(fig, clear_figure=clear_figure)
            if clear_figure is True:
                fig_clf.assert_called_once()
            else:
                fig_clf.assert_not_called()

    @parameterized.expand([(True, image.WidthBehaviour.COLUMN), (False, image.WidthBehaviour.ORIGINAL)])
    def test_st_pyplot_use_container_width(self, use_container_width: bool, image_width: int):
        if False:
            print('Hello World!')
        'st.pyplot should set image width.'
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.hist(np.random.normal(1, 1, size=100), bins=20)
        st.pyplot(fig, use_container_width=use_container_width)
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.imgs.width, image_width)