import os
import tempfile
import unittest
import uuid
from pathlib import Path
from transformers.testing_utils import get_tests_dir, require_soundfile, require_torch, require_vision
from transformers.tools.agent_types import AgentAudio, AgentImage, AgentText
from transformers.utils import is_soundfile_availble, is_torch_available, is_vision_available
if is_torch_available():
    import torch
if is_soundfile_availble():
    import soundfile as sf
if is_vision_available():
    from PIL import Image

def get_new_path(suffix='') -> str:
    if False:
        for i in range(10):
            print('nop')
    directory = tempfile.mkdtemp()
    return os.path.join(directory, str(uuid.uuid4()) + suffix)

@require_soundfile
@require_torch
class AgentAudioTests(unittest.TestCase):

    def test_from_tensor(self):
        if False:
            return 10
        tensor = torch.rand(12, dtype=torch.float64) - 0.5
        agent_type = AgentAudio(tensor)
        path = str(agent_type.to_string())
        self.assertTrue(torch.allclose(tensor, agent_type.to_raw(), atol=0.0001))
        del agent_type
        self.assertTrue(os.path.exists(path))
        (new_tensor, _) = sf.read(path)
        self.assertTrue(torch.allclose(tensor, torch.tensor(new_tensor), atol=0.0001))

    def test_from_string(self):
        if False:
            print('Hello World!')
        tensor = torch.rand(12, dtype=torch.float64) - 0.5
        path = get_new_path(suffix='.wav')
        sf.write(path, tensor, 16000)
        agent_type = AgentAudio(path)
        self.assertTrue(torch.allclose(tensor, agent_type.to_raw(), atol=0.0001))
        self.assertEqual(agent_type.to_string(), path)

@require_vision
@require_torch
class AgentImageTests(unittest.TestCase):

    def test_from_tensor(self):
        if False:
            i = 10
            return i + 15
        tensor = torch.randint(0, 256, (64, 64, 3))
        agent_type = AgentImage(tensor)
        path = str(agent_type.to_string())
        self.assertTrue(torch.allclose(tensor, agent_type._tensor, atol=0.0001))
        self.assertIsInstance(agent_type.to_raw(), Image.Image)
        del agent_type
        self.assertTrue(os.path.exists(path))

    def test_from_string(self):
        if False:
            while True:
                i = 10
        path = Path(get_tests_dir('fixtures/tests_samples/COCO')) / '000000039769.png'
        image = Image.open(path)
        agent_type = AgentImage(path)
        self.assertTrue(path.samefile(agent_type.to_string()))
        self.assertTrue(image == agent_type.to_raw())
        del agent_type
        self.assertTrue(os.path.exists(path))

    def test_from_image(self):
        if False:
            for i in range(10):
                print('nop')
        path = Path(get_tests_dir('fixtures/tests_samples/COCO')) / '000000039769.png'
        image = Image.open(path)
        agent_type = AgentImage(image)
        self.assertFalse(path.samefile(agent_type.to_string()))
        self.assertTrue(image == agent_type.to_raw())
        del agent_type
        self.assertTrue(os.path.exists(path))

class AgentTextTests(unittest.TestCase):

    def test_from_string(self):
        if False:
            return 10
        string = 'Hey!'
        agent_type = AgentText(string)
        self.assertEqual(string, agent_type.to_string())
        self.assertEqual(string, agent_type.to_raw())
        self.assertEqual(string, agent_type)