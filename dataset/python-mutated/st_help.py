import streamlit as st
from streamlit.elements.doc_string import _get_scriptrunner_frame
if _get_scriptrunner_frame() is None:
    st.warning("\n        You're running this script in an `exec` context, so the `foo` part\n        of `st.help(foo)` will not appear inside the displayed `st.help` element.\n        ")

class FooWithNoDocs:
    my_static_var_1 = 123
st.help(FooWithNoDocs)
st.help(globals)

class FooWithLongDocs:
    """My docstring.

    This is a very long one! You probably need to scroll, scroll, scroll, scroll,
    scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll,
    scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll,
    scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll,
    scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll,
    scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll.

    Scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll,
    scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll,
    scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll,
    scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll.

    Scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll,
    scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll,
    scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll,
    scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll.

    Scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll,
    scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll,
    scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll,
    scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll.

    Scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll,
    scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll,
    scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll,
    scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll, scroll.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.my_var_1 = 123

    def my_func_1(self, a, b=False):
        if False:
            print('Hello World!')
        'Func with doc.'

    def my_func_2(self):
        if False:
            return 10
        pass
f = FooWithLongDocs()
f