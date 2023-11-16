from pandasai.responses.response_parser import ResponseParser

class StreamlitResponse(ResponseParser):

    def __init__(self, context):
        if False:
            i = 10
            return i + 15
        super().__init__(context)

    def format_plot(self, result) -> None:
        if False:
            print('Hello World!')
        '\n        Display plot against a user query in Streamlit\n        Args:\n            result (dict): result contains type and value\n        '
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        try:
            image = mpimg.imread(result['value'])
        except FileNotFoundError as e:
            raise FileNotFoundError(f"The file {result['value']} does not exist.") from e
        except OSError as e:
            raise ValueError(f"The file {result['value']} is not a valid image file.") from e
        try:
            import streamlit as st
        except ImportError as exc:
            raise ImportError("The 'streamlit' module is required to use StreamLit Response. Please install it using pip: pip install streamlit") from exc
        plt.imshow(image)
        plt.axis('off')
        fig = plt.gcf()
        st.pyplot(fig)