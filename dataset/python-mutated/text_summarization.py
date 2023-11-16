from ..models.auto import AutoModelForSeq2SeqLM, AutoTokenizer
from .base import PipelineTool

class TextSummarizationTool(PipelineTool):
    """
    Example:

    ```py
    from transformers.tools import TextSummarizationTool

    summarizer = TextSummarizationTool()
    summarizer(long_text)
    ```
    """
    default_checkpoint = 'philschmid/bart-large-cnn-samsum'
    description = 'This is a tool that summarizes an English text. It takes an input `text` containing the text to summarize, and returns a summary of the text.'
    name = 'summarizer'
    pre_processor_class = AutoTokenizer
    model_class = AutoModelForSeq2SeqLM
    inputs = ['text']
    outputs = ['text']

    def encode(self, text):
        if False:
            return 10
        return self.pre_processor(text, return_tensors='pt', truncation=True)

    def forward(self, inputs):
        if False:
            print('Hello World!')
        return self.model.generate(**inputs)[0]

    def decode(self, outputs):
        if False:
            for i in range(10):
                print('nop')
        return self.pre_processor.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)