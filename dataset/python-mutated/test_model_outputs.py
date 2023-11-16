import unittest
import torch
from modelscope.outputs import TextClassificationModelOutput
from modelscope.utils.test_utils import test_level

class TestModelOutput(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        pass

    def tearDown(self):
        if False:
            print('Hello World!')
        super().tearDown()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_model_outputs(self):
        if False:
            for i in range(10):
                print('nop')
        outputs = TextClassificationModelOutput(logits=torch.Tensor([1]))
        self.assertEqual(outputs['logits'], torch.Tensor([1]))
        self.assertEqual(outputs[0], torch.Tensor([1]))
        self.assertEqual(outputs.logits, torch.Tensor([1]))
        outputs.loss = torch.Tensor([2])
        (logits, loss) = outputs
        self.assertEqual(logits, torch.Tensor([1]))
        self.assertTrue(loss is not None)
if __name__ == '__main__':
    unittest.main()