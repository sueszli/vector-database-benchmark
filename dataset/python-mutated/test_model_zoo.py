import itertools
import os
import subprocess
from os.path import dirname
from parameterized import parameterized
from tests.trainer.test_trainer import TrainerIntegrationCommon
from transformers import is_torch_available
from transformers.testing_utils import TestCasePlus, execute_subprocess_async, get_gpu_count, get_tests_dir, require_deepspeed, require_torch_gpu, slow
from transformers.trainer_utils import set_seed
if is_torch_available():
    from tests.trainer.test_trainer import RegressionModelConfig, RegressionPreTrainedModel, get_regression_trainer
set_seed(42)
FIXTURE_DIRECTORY = get_tests_dir('fixtures')
ROOT_DIRECTORY = os.path.join(dirname(get_tests_dir()))
DS_TESTS_DIRECTORY = dirname(os.path.abspath(__file__))
DEFAULT_MASTER_PORT = '10999'
T5_SMALL = 't5-small'
ALBERT_TINY = 'hf-internal-testing/tiny-albert'
BART_TINY = 'sshleifer/bart-tiny-random'
BERT_TINY = 'hf-internal-testing/tiny-bert'
BIGBIRD_PEGASUS_TINY = 'hf-internal-testing/tiny-random-bigbird_pegasus'
BIG_BIRD_TINY = 'hf-internal-testing/tiny-random-big_bird'
BLENDERBOT_TINY = 'hf-internal-testing/tiny-random-blenderbot'
BLOOM_TINY = 'bigscience/bigscience-small-testing'
DEBERTA_TINY = 'hf-internal-testing/tiny-random-deberta'
DEBERTA_V2_TINY = 'hf-internal-testing/tiny-random-deberta-v2'
DISTILBERT_TINY = 'sshleifer/tiny-distilbert-base-cased'
ELECTRA_TINY = 'hf-internal-testing/tiny-electra'
FLAUBERT_TINY = 'hf-internal-testing/tiny-random-flaubert'
FSMT_TINY = 'stas/tiny-wmt19-en-de'
FUNNEL_TINY = 'hf-internal-testing/tiny-random-funnel'
GPT2_TINY = 'sshleifer/tiny-gpt2'
GPTJ_TINY = 'hf-internal-testing/tiny-random-gptj'
GPT_NEO_TINY = 'hf-internal-testing/tiny-random-gpt_neo'
LAYOUTLM_TINY = 'hf-internal-testing/tiny-layoutlm'
LED_TINY = 'hf-internal-testing/tiny-random-led'
LONGFORMER_TINY = 'hf-internal-testing/tiny-random-longformer'
M2M_100_TINY = 'stas/tiny-m2m_100'
MARIAN_TINY = 'sshleifer/tiny-marian-en-de'
MBART_TINY = 'sshleifer/tiny-mbart'
MOBILEBERT_TINY = 'hf-internal-testing/tiny-random-mobilebert'
MPNET_TINY = 'hf-internal-testing/tiny-random-mpnet'
PEGASUS_TINY = 'stas/pegasus-cnn_dailymail-tiny-random'
PROPHETNET_TINY = 'hf-internal-testing/tiny-random-prophetnet'
ROBERTA_TINY = 'sshleifer/tiny-distilroberta-base'
SQUEEZEBERT_TINY = 'hf-internal-testing/tiny-random-squeezebert'
T5_TINY = 'patrickvonplaten/t5-tiny-random'
T5_V1_TINY = 'hf-internal-testing/tiny-random-t5-v1.1'
VIT_TINY = 'hf-internal-testing/tiny-random-vit'
XLM_ROBERTA_TINY = 'hf-internal-testing/tiny-xlm-roberta'
XLNET_TINY = 'sshleifer/tiny-xlnet-base-cased'
MT5_TINY = 'hf-internal-testing/tiny-random-mt5'
CAMEMBERT_TINY = 'hf-internal-testing/tiny-random-camembert'
OPENAI_GPT_TINY = 'hf-internal-testing/tiny-random-openai-gpt'
CONVBERT_TINY = 'hf-internal-testing/tiny-random-convbert'
LAYOUTLMV2_TINY = 'hf-internal-testing/tiny-random-layoutlmv2'
HUBERT_TINY = 'hf-internal-testing/tiny-random-hubert'
CTRL_TINY = 'hf-internal-testing/tiny-random-ctrl'
TRANSFO_XL_TINY = 'hf-internal-testing/tiny-random-transfo-xl'
IBERT_TINY = 'hf-internal-testing/tiny-random-ibert'
REFORMER_TINY = 'hf-internal-testing/tiny-random-reformer'
DPR_TINY = 'hf-internal-testing/tiny-random-dpr'
RAG_TINY = 'hf-internal-testing/tiny-random-rag'
LUKE_TINY = ''
LXMERT_TINY = 'hf-internal-testing/tiny-random-lxmert'
CLIP_TINY = 'hf-internal-testing/tiny-random-clip'
SPEECH_TO_TEXT_TINY = 'hf-internal-testing/tiny-random-speech_to_text'
TAPAS_TINY = 'hf-internal-testing/tiny-random-tapas'

def get_launcher(distributed=False):
    if False:
        while True:
            i = 10
    num_gpus = min(2, get_gpu_count()) if distributed else 1
    master_port = os.environ.get('DS_TEST_PORT', DEFAULT_MASTER_PORT)
    return f'deepspeed --num_nodes 1 --num_gpus {num_gpus} --master_port {master_port}'.split()

def make_task_cmds():
    if False:
        return 10
    data_dir_samples = f'{FIXTURE_DIRECTORY}/tests_samples'
    data_dir_wmt = f'{data_dir_samples}/wmt_en_ro'
    data_dir_xsum = f'{data_dir_samples}/xsum'
    args_main = '\n        --do_train\n        --max_train_samples 4\n        --per_device_train_batch_size 2\n        --num_train_epochs 1\n        --fp16\n        --report_to none\n        --overwrite_output_dir\n        '.split()
    tasks2models = {'trans': ['bart', 'fsmt', 'm2m_100', 'marian', 'mbart', 't5', 't5_v1'], 'sum': ['pegasus'], 'clm': ['big_bird', 'bigbird_pegasus', 'blenderbot', 'bloom', 'gpt2', 'gpt_neo', 'gptj', 'xlm-roberta', 'prophetnet'], 'mlm': ['albert', 'deberta', 'deberta-v2', 'distilbert', 'electra', 'flaubert', 'funnel', 'layoutlm'], 'qa': ['led', 'longformer', 'mobilebert', 'mpnet', 'roberta', 'squeezebert'], 'clas': ['bert', 'xlnet'], 'img_clas': ['vit']}
    scripts_dir = f'{ROOT_DIRECTORY}/examples/pytorch'
    tasks = {'trans': f'\n        {scripts_dir}/translation/run_translation.py\n        --train_file {data_dir_wmt}/train.json\n        --source_lang en\n        --target_lang ro\n        ', 'sum': f'\n        {scripts_dir}/summarization/run_summarization.py\n        --train_file {data_dir_xsum}/sample.json\n        --max_source_length 12\n        --max_target_length 12\n        --lang en\n        ', 'clm': f'\n        {scripts_dir}/language-modeling/run_clm.py\n        --train_file {FIXTURE_DIRECTORY}/sample_text.txt\n        --block_size 8\n        ', 'mlm': f'\n        {scripts_dir}/language-modeling/run_mlm.py\n        --train_file {FIXTURE_DIRECTORY}/sample_text.txt\n        ', 'qa': f'\n        {scripts_dir}/question-answering/run_qa.py\n        --train_file {data_dir_samples}/SQUAD/sample.json\n        ', 'clas': f'\n        {scripts_dir}/text-classification/run_glue.py\n        --train_file {data_dir_samples}/MRPC/train.csv\n        --max_seq_length 12\n        --task_name MRPC\n        ', 'img_clas': f'\n        {scripts_dir}/image-classification/run_image_classification.py\n            --dataset_name hf-internal-testing/cats_vs_dogs_sample\n            --remove_unused_columns False\n            --max_steps 10\n            --image_processor_name {DS_TESTS_DIRECTORY}/vit_feature_extractor.json\n        '}
    launcher = get_launcher(distributed=True)
    cmds = {}
    for (task, args) in tasks.items():
        args = args.split()
        for model in tasks2models[task]:
            model_name = globals()[f"{model.upper().replace('-', '_')}_TINY"]
            args_model = f'--model_name_or_path {model_name}'.split()
            cmds[f'{task}_{model}'] = launcher + args + args_model + args_main
    return cmds
task_cmds = make_task_cmds()
ZERO2 = 'zero2'
ZERO3 = 'zero3'
stages = [ZERO2, ZERO3]

def parameterized_custom_name_func(func, param_num, param):
    if False:
        return 10
    param_based_name = parameterized.to_safe_name('_'.join((str(x) for x in param.args)))
    return f'{func.__name__}_{param_based_name}'
params = list(itertools.product(stages, task_cmds.keys()))

@slow
@require_deepspeed
@require_torch_gpu
class TestDeepSpeedModelZoo(TestCasePlus):
    """This class is for testing via an external script - can do multiple gpus"""

    def get_task_cmd(self, task, stage):
        if False:
            for i in range(10):
                print('nop')
        if task not in task_cmds:
            raise ValueError(f"don't know of task {task}, have {task_cmds.keys()}")
        cmd = task_cmds[task]
        args_ds = f'--deepspeed {self.test_file_dir_str}/ds_config_{stage}.json'.split()
        output_dir = self.get_auto_remove_tmp_dir()
        args_out = f'--output_dir {output_dir}'.split()
        cmd += args_ds + args_out
        return (cmd, output_dir)

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_zero_to_fp32(self, stage, task):
        if False:
            return 10
        (cmd, output_dir) = self.get_task_cmd(task, stage)
        cmd += '--save_steps 1'.split()
        execute_subprocess_async(cmd, env=self.get_env())
        chkpt_dir = f'{output_dir}/checkpoint-1'
        recovered_model_path = f'{chkpt_dir}/out.bin'
        cmd = f'{chkpt_dir}/zero_to_fp32.py {chkpt_dir} {recovered_model_path}'
        subprocess.check_call(cmd, shell=True)
        assert os.path.exists(recovered_model_path), f'{recovered_model_path} was not found'