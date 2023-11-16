import os
from lightning.app import LightningFlow, LightningApp
from lightning.app.frontend import StaticWebFrontend, StreamlitFrontend
from lightning.app.utilities.state import AppState

class UIStreamLit(LightningFlow):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.should_print = False

    def configure_layout(self):
        if False:
            i = 10
            return i + 15
        return StreamlitFrontend(render_fn=render_fn)

def render_fn(state: AppState):
    if False:
        i = 10
        return i + 15
    import streamlit as st
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=2000, limit=None, key='refresh')
    state.should_print = st.select_slider("Should the Application print 'Hello World !' to the terminal:", [False, True])

class UIStatic(LightningFlow):

    def configure_layout(self):
        if False:
            return 10
        return StaticWebFrontend(os.path.join(os.path.dirname(__file__), 'ui'))

class HelloWorld(LightningFlow):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.static_ui = UIStatic()
        self.streamlit_ui = UIStreamLit()

    def run(self):
        if False:
            return 10
        print('Hello World!' if self.streamlit_ui.should_print else '')

    def configure_layout(self):
        if False:
            print('Hello World!')
        return [{'name': 'StreamLit', 'content': self.streamlit_ui}, {'name': 'Static', 'content': self.static_ui}]
app = LightningApp(HelloWorld())