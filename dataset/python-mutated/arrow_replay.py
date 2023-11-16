"""A small script that reproduces the bug from #6103"""
import pandas as pd
import streamlit as st
df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': ['pino', 'gino', 'lucullo', 'augusto']})
st.write(df)

@st.cache_resource(show_spinner=False)
def stampa_df(df_param: pd.DataFrame) -> None:
    if False:
        while True:
            i = 10
    st.write(df_param)
stampa_df(df)