"""
Basic similarity search example. Used in the original txtai demo.

Requires streamlit to be installed.
  pip install streamlit
"""
import os
import streamlit as st
from txtai.embeddings import Embeddings

class Application:
    """
    Main application.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        '\n        Creates a new application.\n        '
        self.embeddings = Embeddings({'path': 'sentence-transformers/nli-mpnet-base-v2'})

    def run(self):
        if False:
            i = 10
            return i + 15
        '\n        Runs a Streamlit application.\n        '
        st.title('Similarity Search')
        st.markdown('This application runs a basic similarity search that identifies the best matching row for a query.')
        data = ['US tops 5 million confirmed virus cases', "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg", 'Beijing mobilises invasion craft along coast as Taiwan tensions escalate', 'The National Park Service warns against sacrificing slower friends in a bear attack', 'Maine man wins $1M from $25 lottery ticket', 'Make huge profits without work, earn up to $100,000 a day']
        data = st.text_area('Data', value='\n'.join(data))
        query = st.text_input('Query')
        data = data.split('\n')
        if query:
            uid = self.embeddings.similarity(query, data)[0][0]
            st.write(data[uid])

@st.cache(allow_output_mutation=True)
def create():
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates and caches a Streamlit application.\n\n    Returns:\n        Application\n    '
    return Application()
if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    app = create()
    app.run()