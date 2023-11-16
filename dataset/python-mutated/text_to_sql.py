import os
from typing import Dict, Optional
import torch
from text2sql_lgesql.asdl.asdl import ASDLGrammar
from text2sql_lgesql.asdl.transition_system import TransitionSystem
from text2sql_lgesql.model.model_constructor import Text2SQL
from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.utils.compatible_with_transformers import compatible_position_ids
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
__all__ = ['StarForTextToSql']

@MODELS.register_module(Tasks.table_question_answering, module_name=Models.space_T_en)
class StarForTextToSql(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        if False:
            print('Hello World!')
        'initialize the star model from the `model_dir` path.\n\n        Args:\n            model_dir (str): the model path.\n        '
        super().__init__(model_dir, *args, **kwargs)
        self.beam_size = 5
        self.config = kwargs.pop('config', Config.from_file(os.path.join(self.model_dir, ModelFile.CONFIGURATION)))
        self.config.model.model_dir = model_dir
        self.grammar = ASDLGrammar.from_filepath(os.path.join(model_dir, 'sql_asdl_v2.txt'))
        self.trans = TransitionSystem.get_class_by_lang('sql')(self.grammar)
        self.arg = self.config.model
        self.device = 'cuda' if ('device' not in kwargs or kwargs['device'] == 'gpu') and torch.cuda.is_available() else 'cpu'
        self.model = Text2SQL(self.arg, self.trans)
        check_point = torch.load(open(os.path.join(model_dir, ModelFile.TORCH_MODEL_BIN_FILE), 'rb'), map_location=self.device)
        compatible_position_ids(check_point['model'], 'encoder.input_layer.plm_model.embeddings.position_ids')
        self.model.load_state_dict(check_point['model'])

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if False:
            i = 10
            return i + 15
        'return the result by the model\n\n        Args:\n            input (Dict[str, Tensor]): the preprocessed data\n\n        Returns:\n            Dict[str, Tensor]: results\n                Example:\n\n        Example:\n            >>> from modelscope.hub.snapshot_download import snapshot_download\n            >>> from modelscope.models.nlp import StarForTextToSql\n            >>> from modelscope.preprocessors import ConversationalTextToSqlPreprocessor\n            >>> test_case = {\n                    \'database_id\': \'employee_hire_evaluation\',\n                    \'local_db_path\': None,\n                    \'utterance\': [\n                        "I\'d like to see Shop names.", \'Which of these are hiring?\',\n                        \'Which shop is hiring the highest number of employees?\'\n                        \' | do you want the name of the shop ? | Yes\'\n                    ]\n                }\n            >>> cache_path = snapshot_download(\'damo/nlp_star_conversational-text-to-sql\')\n            >>> preprocessor = ConversationalTextToSqlPreprocessor(\n                    model_dir=cache_path,\n                    database_id=test_case[\'database_id\'],\n                db_content=True)\n            >>> model = StarForTextToSql(cache_path, config=preprocessor.config)\n            >>> print(model(preprocessor({\n                    \'utterance\': "I\'d like to see Shop names.",\n                    \'history\': [],\n                    \'last_sql\': \'\',\n                    \'database_id\': \'employee_hire_evaluation\',\n                    \'local_db_path\': None\n                })))\n        '
        self.model.eval()
        hyps = self.model.parse(input['batch'], self.beam_size)
        db = input['batch'].examples[0].db
        predict = {'predict': hyps, 'db': db}
        return predict