from typing import Any, Dict, Union
from transformers import AutoTokenizer
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, InputFields

@PREPROCESSORS.register_module(Fields.nlp)
class Tokenize(Preprocessor):

    def __init__(self, tokenizer_name) -> None:
        if False:
            i = 10
            return i + 15
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if False:
            return 10
        if isinstance(data, str):
            data = {InputFields.text: data}
        token_dict = self.tokenizer(data[InputFields.text])
        data.update(token_dict)
        return data