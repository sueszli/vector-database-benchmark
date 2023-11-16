from typing import Any, Dict, Optional, Union
import torch
from modelscope.metainfo import Pipelines
from modelscope.models.multi_modal.vldoc.model import VLDocForDocVLEmbedding
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.multi_modal import Preprocessor, VLDocPreprocessor
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.document_vl_embedding, module_name=Pipelines.document_vl_embedding)
class DocumentVLEmbeddingPipeline(Pipeline):

    def __init__(self, model: Union[Model, str], preprocessor: Optional[Preprocessor]=None, **kwargs):
        if False:
            i = 10
            return i + 15
        " The pipeline for multi-modal document embedding generation.\n\n        Args:\n            model: model id on modelscope hub.\n            preprocessor: type `Preprocessor`. If None, `VLDocPreprocessor` is used.\n\n        Examples:\n\n        >>> from modelscope.models import Model\n        >>> from modelscope.pipelines import pipeline\n        >>> model = Model.from_pretrained(\n            'damo/multi-modal_convnext-roberta-base_vldoc-embedding')\n        >>> doc_VL_emb_pipeline = pipeline(task='document-vl-embedding', model=model)\n        >>> inp = {\n                'images': ['data/demo.png'],\n                'ocr_info_paths': ['data/demo.json']\n            }\n        >>> result = doc_VL_emb_pipeline(inp)\n        "
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.model.eval()
        if preprocessor is None:
            if isinstance(self.model, VLDocForDocVLEmbedding):
                self.preprocessor = VLDocPreprocessor(self.model.model_dir)
            else:
                raise NotImplementedError

    def forward(self, encodings: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        for (k, v) in encodings.items():
            encodings[k] = encodings[k].to(self.device)
        return self.model(**encodings)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return inputs