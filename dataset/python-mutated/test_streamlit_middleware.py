"""Unit tests for the streamlit middleware class"""
import pandas as pd
from pandasai.smart_datalake import SmartDatalake
from pandasai.llm.fake import FakeLLM
from pandasai.middlewares import StreamlitMiddleware

class TestStreamlitMiddleware:
    """Unit tests for the streamlit middleware class"""

    def test_streamlit_middleware(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the streamlit middleware'
        code = 'plt.show()'
        middleware = StreamlitMiddleware()
        assert middleware(code=code) == 'import streamlit as st\nst.pyplot(plt.gcf())'
        assert middleware.has_run

    def test_streamlit_middleware_optional_dependency(self, mock_json_load):
        if False:
            while True:
                i = 10
        'Test the streamlit middleware installs the optional dependency'
        mock_json_load.return_value = {}
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        llm = FakeLLM('import matplotlib.pyplot as plt\ndef analyze_data(dfs):\n    return { \'type\': \'text\', \'value\': "Hello World" }')
        dl = SmartDatalake([df], config={'llm': llm, 'middlewares': [StreamlitMiddleware()], 'enable_cache': False})
        dl.chat('Plot the histogram of countries showing for each the gpd, using differentcolors for each bar')
        assert {'module': 'streamlit', 'name': 'streamlit', 'alias': 'st'} in dl._code_manager._additional_dependencies