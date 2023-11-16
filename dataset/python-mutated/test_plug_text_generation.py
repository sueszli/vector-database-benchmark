import unittest
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

class TextPlugGenerationTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.model_id = 'damo/nlp_plug_text-generation_27B'
        self.model_dir = snapshot_download(self.model_id)
        self.plug_input = '段誉轻挥折扇，摇了摇头，说道：“你师父是你的师父，你师父可不是我的师父。"'

    @unittest.skip('distributed plug, skipped')
    def test_plug(self):
        if False:
            print('Hello World!')
        ' The model can be downloaded from the link on\n        https://modelscope.cn/models/damo/nlp_plug_text-generation_27B/summary.\n        After downloading, you should have a plug model structure like this:\n        nlp_plug_text-generation_27B\n            |_ config.json\n            |_ configuration.json\n            |_ ds_zero-offload_10B_config.json\n            |_ vocab.txt\n            |_ model <-- an empty directory\n\n        Model binaries shall be downloaded separately to populate the model directory, so that\n        the model directory would contain the following binaries:\n            |_ model\n                |_ mp_rank_00_model_states.pt\n                |_ mp_rank_01_model_states.pt\n                |_ mp_rank_02_model_states.pt\n                |_ mp_rank_03_model_states.pt\n                |_ mp_rank_04_model_states.pt\n                |_ mp_rank_05_model_states.pt\n                |_ mp_rank_06_model_states.pt\n                |_ mp_rank_07_model_states.pt\n        '
        pipe = pipeline(Tasks.text_generation, model=self.model_id)
        print(f'input: {self.plug_input}\noutput: {pipe(self.plug_input, out_length=256)}')
if __name__ == '__main__':
    unittest.main()