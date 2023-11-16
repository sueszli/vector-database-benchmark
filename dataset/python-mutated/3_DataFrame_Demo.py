from urllib.error import URLError
import altair as alt
import pandas as pd
import streamlit as st
from streamlit.hello.utils import show_code

def data_frame_demo():
    if False:
        while True:
            i = 10

    @st.cache_data
    def get_UN_data():
        if False:
            return 10
        AWS_BUCKET_URL = 'https://streamlit-demo-data.s3-us-west-2.amazonaws.com'
        df = pd.read_csv(AWS_BUCKET_URL + '/agri.csv.gz')
        return df.set_index('Region')
    try:
        df = get_UN_data()
        countries = st.multiselect('Choose countries', list(df.index), ['China', 'United States of America'])
        if not countries:
            st.error('Please select at least one country.')
        else:
            data = df.loc[countries]
            data /= 1000000.0
            st.write('### Gross Agricultural Production ($B)', data.sort_index())
            data = data.T.reset_index()
            data = pd.melt(data, id_vars=['index']).rename(columns={'index': 'year', 'value': 'Gross Agricultural Product ($B)'})
            chart = alt.Chart(data).mark_area(opacity=0.3).encode(x='year:T', y=alt.Y('Gross Agricultural Product ($B):Q', stack=None), color='Region:N')
            st.altair_chart(chart, use_container_width=True)
    except URLError as e:
        st.error('\n            **This demo requires internet access.**\n            Connection error: %s\n        ' % e.reason)
st.set_page_config(page_title='DataFrame Demo', page_icon='📊')
st.markdown('# DataFrame Demo')
st.sidebar.header('DataFrame Demo')
st.write('This demo shows how to use `st.write` to visualize Pandas DataFrames.\n(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)')
data_frame_demo()
show_code(data_frame_demo)