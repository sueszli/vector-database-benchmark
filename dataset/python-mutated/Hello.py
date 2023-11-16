import streamlit as st
from streamlit.logger import get_logger
LOGGER = get_logger(__name__)

def run():
    if False:
        i = 10
        return i + 15
    st.set_page_config(page_title='Hello', page_icon='👋')
    st.write('# Welcome to Streamlit! 👋')
    st.sidebar.success('Select a demo above.')
    st.markdown('\n        Streamlit is an open-source app framework built specifically for\n        Machine Learning and Data Science projects.\n        **👈 Select a demo from the sidebar** to see some examples\n        of what Streamlit can do!\n        ### Want to learn more?\n        - Check out [streamlit.io](https://streamlit.io)\n        - Jump into our [documentation](https://docs.streamlit.io)\n        - Ask a question in our [community\n          forums](https://discuss.streamlit.io)\n        ### See more complex demos\n        - Use a neural net to [analyze the Udacity Self-driving Car Image\n          Dataset](https://github.com/streamlit/demo-self-driving)\n        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)\n    ')
if __name__ == '__main__':
    run()