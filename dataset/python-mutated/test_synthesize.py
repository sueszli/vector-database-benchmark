import os
from tests import get_tests_output_path, run_cli

def test_synthesize():
    if False:
        for i in range(10):
            print('nop')
    'Test synthesize.py with diffent arguments.'
    output_path = os.path.join(get_tests_output_path(), 'output.wav')
    run_cli('tts --list_models')
    run_cli(f'tts --text "This is an example." --out_path "{output_path}"')
    run_cli(f'tts --model_name tts_models/en/ljspeech/glow-tts --text "This is an example." --out_path "{output_path}"')
    run_cli(f'tts --model_name tts_models/en/ljspeech/glow-tts  --vocoder_name vocoder_models/en/ljspeech/multiband-melgan --text "This is an example." --out_path "{output_path}"')