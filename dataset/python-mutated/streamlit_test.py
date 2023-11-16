"""Streamlit Unit test."""
import os
import re
import subprocess
import sys
import tempfile
import unittest
import streamlit as st
from streamlit import __version__

def get_version():
    if False:
        while True:
            i = 10
    'Get version by parsing out setup.py.'
    dirname = os.path.dirname(__file__)
    base_dir = os.path.abspath(os.path.join(dirname, '../..'))
    pattern = re.compile('(?:.*VERSION = \\")(?P<version>.*)(?:\\"  # PEP-440$)')
    for line in open(os.path.join(base_dir, 'setup.py')).readlines():
        m = pattern.match(line)
        if m:
            return m.group('version')

class StreamlitTest(unittest.TestCase):
    """Test Streamlit.__init__.py."""

    def test_streamlit_version(self):
        if False:
            i = 10
            return i + 15
        'Test streamlit.__version__.'
        self.assertEqual(__version__, get_version())

    def test_get_option(self):
        if False:
            return 10
        'Test streamlit.get_option.'
        self.assertEqual(False, st.get_option('browser.gatherUsageStats'))

    def test_public_api(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that we don't accidentally remove (or add) symbols\n        to the public `st` API.\n        "
        api = {k for (k, v) in st.__dict__.items() if not k.startswith('_') and (not isinstance(v, type(st)))}
        self.assertEqual(api, {'altair_chart', 'area_chart', 'audio', 'balloons', 'bar_chart', 'bokeh_chart', 'button', 'caption', 'camera_input', 'chat_input', 'chat_message', 'checkbox', 'code', 'columns', 'tabs', 'container', 'dataframe', 'data_editor', 'date_input', 'divider', 'download_button', 'expander', 'pydeck_chart', 'empty', 'error', 'exception', 'file_uploader', 'form', 'form_submit_button', 'graphviz_chart', 'header', 'help', 'image', 'info', 'json', 'latex', 'line_chart', 'link_button', 'map', 'markdown', 'metric', 'multiselect', 'number_input', 'plotly_chart', 'progress', 'pyplot', 'radio', 'scatter_chart', 'selectbox', 'select_slider', 'slider', 'snow', 'subheader', 'success', 'status', 'table', 'text', 'text_area', 'text_input', 'time_input', 'title', 'toast', 'toggle', 'vega_lite_chart', 'video', 'warning', 'write', 'color_picker', 'sidebar', 'event', 'echo', 'spinner', 'set_page_config', 'stop', 'rerun', 'cache', 'secrets', 'session_state', 'cache_data', 'cache_resource', 'experimental_user', 'experimental_singleton', 'experimental_memo', 'experimental_get_query_params', 'experimental_set_query_params', 'experimental_rerun', 'experimental_data_editor', 'experimental_connection', 'get_option', 'set_option', 'connection'})

    def test_pydoc(self):
        if False:
            print('Hello World!')
        'Test that we can run pydoc on the streamlit package'
        cwd = os.getcwd()
        try:
            os.chdir(tempfile.mkdtemp())
            output = subprocess.check_output([sys.executable, '-m', 'pydoc', 'streamlit']).decode()
            self.assertIn('Help on package streamlit:', output)
        finally:
            os.chdir(cwd)