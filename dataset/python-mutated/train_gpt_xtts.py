import os
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
RUN_NAME = 'GPT_XTTS_LJSpeech_FT'
PROJECT_NAME = 'XTTS_trainer'
DASHBOARD_LOGGER = 'tensorboard'
LOGGER_URI = None
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run', 'training')
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True
START_WITH_EVAL = True
BATCH_SIZE = 3
GRAD_ACUMM_STEPS = 84
config_dataset = BaseDatasetConfig(formatter='ljspeech', dataset_name='ljspeech', path='/raid/datasets/LJSpeech-1.1_24khz/', meta_file_train='/raid/datasets/LJSpeech-1.1_24khz/metadata.csv', language='en')
DATASETS_CONFIG_LIST = [config_dataset]
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, 'XTTS_v1.1_original_model_files/')
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)
DVAE_CHECKPOINT_LINK = 'https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/dvae.pth'
MEL_NORM_LINK = 'https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/mel_stats.pth'
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, DVAE_CHECKPOINT_LINK.split('/')[-1])
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, MEL_NORM_LINK.split('/')[-1])
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(' > Downloading DVAE files!')
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)
TOKENIZER_FILE_LINK = 'https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/vocab.json'
XTTS_CHECKPOINT_LINK = 'https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/model.pth'
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, TOKENIZER_FILE_LINK.split('/')[-1])
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, XTTS_CHECKPOINT_LINK.split('/')[-1])
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(' > Downloading XTTS v1.1 files!')
    ModelManager._download_model_files([TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)
SPEAKER_REFERENCE = ['./tests/data/ljspeech/wavs/LJ001-0002.wav']
LANGUAGE = config_dataset.language

def main():
    if False:
        while True:
            i = 10
    model_args = GPTArgs(max_conditioning_length=132300, min_conditioning_length=66150, debug_loading_failures=False, max_wav_length=255995, max_text_length=200, mel_norm_file=MEL_NORM_FILE, dvae_checkpoint=DVAE_CHECKPOINT, xtts_checkpoint=XTTS_CHECKPOINT, tokenizer_file=TOKENIZER_FILE, gpt_num_audio_tokens=8194, gpt_start_audio_token=8192, gpt_stop_audio_token=8193)
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    config = GPTTrainerConfig(output_path=OUT_PATH, model_args=model_args, run_name=RUN_NAME, project_name=PROJECT_NAME, run_description='\n            GPT XTTS training\n            ', dashboard_logger=DASHBOARD_LOGGER, logger_uri=LOGGER_URI, audio=audio_config, batch_size=BATCH_SIZE, batch_group_size=48, eval_batch_size=BATCH_SIZE, num_loader_workers=8, eval_split_max_size=256, print_step=50, plot_step=100, log_model_step=1000, save_step=10000, save_n_checkpoints=1, save_checkpoints=True, print_eval=False, optimizer='AdamW', optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS, optimizer_params={'betas': [0.9, 0.96], 'eps': 1e-08, 'weight_decay': 0.01}, lr=5e-06, lr_scheduler='MultiStepLR', lr_scheduler_params={'milestones': [50000 * 18, 150000 * 18, 300000 * 18], 'gamma': 0.5, 'last_epoch': -1}, test_sentences=[{'text': "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.", 'speaker_wav': SPEAKER_REFERENCE, 'language': LANGUAGE}, {'text': "This cake is great. It's so delicious and moist.", 'speaker_wav': SPEAKER_REFERENCE, 'language': LANGUAGE}])
    model = GPTTrainer.init_from_config(config)
    (train_samples, eval_samples) = load_tts_samples(DATASETS_CONFIG_LIST, eval_split=True, eval_split_max_size=config.eval_split_max_size, eval_split_size=config.eval_split_size)
    trainer = Trainer(TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=START_WITH_EVAL, grad_accum_steps=GRAD_ACUMM_STEPS), config, output_path=OUT_PATH, model=model, train_samples=train_samples, eval_samples=eval_samples)
    trainer.fit()
if __name__ == '__main__':
    main()