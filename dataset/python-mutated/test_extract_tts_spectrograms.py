import os
import unittest
import torch
from tests import get_tests_input_path, get_tests_output_path, run_cli
from TTS.config import load_config
from TTS.tts.models import setup_model
torch.manual_seed(1)

class TestExtractTTSSpectrograms(unittest.TestCase):

    @staticmethod
    def test_GlowTTS():
        if False:
            for i in range(10):
                print('nop')
        config_path = os.path.join(get_tests_input_path(), 'test_glow_tts.json')
        checkpoint_path = os.path.join(get_tests_output_path(), 'glowtts.pth')
        output_path = os.path.join(get_tests_output_path(), 'output_extract_tts_spectrograms/')
        c = load_config(config_path)
        model = setup_model(c)
        torch.save({'model': model.state_dict()}, checkpoint_path)
        run_cli(f'CUDA_VISIBLE_DEVICES="" python TTS/bin/extract_tts_spectrograms.py --config_path "{config_path}" --checkpoint_path "{checkpoint_path}" --output_path "{output_path}"')
        run_cli(f'rm -rf "{output_path}" "{checkpoint_path}"')

    @staticmethod
    def test_Tacotron2():
        if False:
            i = 10
            return i + 15
        config_path = os.path.join(get_tests_input_path(), 'test_tacotron2_config.json')
        checkpoint_path = os.path.join(get_tests_output_path(), 'tacotron2.pth')
        output_path = os.path.join(get_tests_output_path(), 'output_extract_tts_spectrograms/')
        c = load_config(config_path)
        model = setup_model(c)
        torch.save({'model': model.state_dict()}, checkpoint_path)
        run_cli(f'CUDA_VISIBLE_DEVICES="" python TTS/bin/extract_tts_spectrograms.py --config_path "{config_path}" --checkpoint_path "{checkpoint_path}" --output_path "{output_path}"')
        run_cli(f'rm -rf "{output_path}" "{checkpoint_path}"')

    @staticmethod
    def test_Tacotron():
        if False:
            for i in range(10):
                print('nop')
        config_path = os.path.join(get_tests_input_path(), 'test_tacotron_config.json')
        checkpoint_path = os.path.join(get_tests_output_path(), 'tacotron.pth')
        output_path = os.path.join(get_tests_output_path(), 'output_extract_tts_spectrograms/')
        c = load_config(config_path)
        model = setup_model(c)
        torch.save({'model': model.state_dict()}, checkpoint_path)
        run_cli(f'CUDA_VISIBLE_DEVICES="" python TTS/bin/extract_tts_spectrograms.py --config_path "{config_path}" --checkpoint_path "{checkpoint_path}" --output_path "{output_path}"')
        run_cli(f'rm -rf "{output_path}" "{checkpoint_path}"')