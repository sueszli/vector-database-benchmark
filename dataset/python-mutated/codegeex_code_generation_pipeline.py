from typing import Any, Dict, Union
from modelscope.metainfo import Pipelines
from modelscope.models.nlp import CodeGeeXForCodeGeneration
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import Tasks

@PIPELINES.register_module(group_key=Tasks.code_generation, module_name=Pipelines.codegeex_code_generation)
class CodeGeeXCodeGenerationPipeline(Pipeline):

    def __init__(self, model: Union[CodeGeeXForCodeGeneration, str], preprocessor: [Preprocessor]=None, *args, **kwargs):
        if False:
            return 10
        model = CodeGeeXForCodeGeneration(model) if isinstance(model, str) else model
        self.model = model
        self.model.eval()
        self.model.half()
        self.model.cuda()
        super().__init__(model=model, **kwargs)

    def preprocess(self, inputs, **preprocess_params) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return inputs

    def forward(self, inputs: Union[Dict], **forward_params) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        for para in ['prompt', 'language']:
            if para not in inputs:
                raise Exception('Please check your input format.')
        if inputs['language'] not in ['C++', 'C', 'C#', 'Cuda', 'Objective-C', 'Objective-C++', 'Python', 'Java', 'Scala', 'TeX', 'HTML', 'PHP', 'JavaScript', 'TypeScript', 'Go', 'Shell', 'Rust', 'CSS', 'SQL', 'Kotlin', 'Pascal', 'R', 'Fortran', 'Lean']:
            raise Exception('Make sure the language is in ["C++","C","C#","Cuda","Objective-C","Objective-C++","Python","Java","Scala","TeX","HTML","PHP","JavaScript","TypeScript","Go","Shell","Rust","CSS","SQL","Kotlin","Pascal","R","Fortran","Lean"]')
        return self.model(inputs)

    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        if False:
            return 10
        return input