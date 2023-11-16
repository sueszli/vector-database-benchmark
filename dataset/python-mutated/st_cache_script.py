"""A script for ScriptRunnerTest that uses st.cache"""
import streamlit as st

@st.cache(suppress_st_warning=True)
def cached1():
    if False:
        for i in range(10):
            print('nop')
    st.text('cached function called')
    return 'cached value'

@st.cache(suppress_st_warning=True)
def cached2():
    if False:
        while True:
            i = 10
    st.text('cached function called')
    return 'cached value'

@st.cache(suppress_st_warning=True)
def cached_depending_on_not_yet_defined():
    if False:
        while True:
            i = 10
    st.text('cached_depending_on_not_yet_defined called')
    return depended_on()

def depended_on():
    if False:
        print('Hello World!')
    return 'cached value'

def outer_func():
    if False:
        return 10

    @st.cache(suppress_st_warning=True)
    def cached1():
        if False:
            print('Hello World!')
        st.text('cached function called')
        return 'cached value'

    @st.cache(suppress_st_warning=True)
    def cached2():
        if False:
            for i in range(10):
                print('nop')
        st.text('cached function called')
        return 'cached value'
    cached1()
    cached2()
cached1()
cached2()
outer_func()
cached_depending_on_not_yet_defined()