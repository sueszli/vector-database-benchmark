import os
from tests import get_tests_output_path, run_cli

def test_synthesize():
    if False:
        while True:
            i = 10
    'Test synthesize.py with diffent arguments.'
    output_path = os.path.join(get_tests_output_path(), 'output.wav')
    run_cli(f'tts --model_name "coqui_studio/en/Torcull Diarmuid/coqui_studio" --text "This is it" --out_path "{output_path}"')
    run_cli(f'tts --model_name "coqui_studio/en/Torcull Diarmuid/coqui_studio" --text "This is it but slow" --speed 0.1--out_path "{output_path}"')
    run_cli(f'tts --text "test." --pipe_out --out_path "{output_path}" | aplay')