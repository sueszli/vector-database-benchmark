import glob
import os
import shutil
from tests import get_device_id, get_tests_output_path, run_cli
from TTS.config.shared_configs import BaseAudioConfig
from TTS.encoder.configs.speaker_encoder_config import SpeakerEncoderConfig

def run_test_train():
    if False:
        return 10
    command = f"CUDA_VISIBLE_DEVICES='{get_device_id()}' python TTS/bin/train_encoder.py --config_path {config_path} --coqpit.output_path {output_path} --coqpit.datasets.0.formatter ljspeech_test --coqpit.datasets.0.meta_file_train metadata.csv --coqpit.datasets.0.meta_file_val metadata.csv --coqpit.datasets.0.path tests/data/ljspeech "
    run_cli(command)
config_path = os.path.join(get_tests_output_path(), 'test_speaker_encoder_config.json')
output_path = os.path.join(get_tests_output_path(), 'train_outputs')
config = SpeakerEncoderConfig(batch_size=4, num_classes_in_batch=4, num_utter_per_class=2, eval_num_classes_in_batch=4, eval_num_utter_per_class=2, num_loader_workers=1, epochs=1, print_step=1, save_step=2, print_eval=True, run_eval=True, audio=BaseAudioConfig(num_mels=80))
config.audio.do_trim_silence = True
config.audio.trim_db = 60
config.loss = 'ge2e'
config.save_json(config_path)
print(config)
run_test_train()
continue_path = max(glob.glob(os.path.join(output_path, '*/')), key=os.path.getmtime)
command_train = f"CUDA_VISIBLE_DEVICES='{get_device_id()}' python TTS/bin/train_encoder.py --continue_path {continue_path} "
run_cli(command_train)
shutil.rmtree(continue_path)
config.model_params['model_name'] = 'resnet'
config.save_json(config_path)
run_test_train()
continue_path = max(glob.glob(os.path.join(output_path, '*/')), key=os.path.getmtime)
command_train = f"CUDA_VISIBLE_DEVICES='{get_device_id()}' python TTS/bin/train_encoder.py --continue_path {continue_path} "
run_cli(command_train)
shutil.rmtree(continue_path)
config.loss = 'softmaxproto'
config.save_json(config_path)
run_test_train()