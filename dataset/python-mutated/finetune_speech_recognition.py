import os
from modelscope.metainfo import Trainers
from modelscope.msdatasets.dataset_cls.custom_datasets import ASRDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode

def modelscope_finetune(params):
    if False:
        return 10
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir, exist_ok=True)
    ds_dict = ASRDataset.load(params.data_path, namespace='speech_asr', download_mode=params.download_mode)
    kwargs = dict(model=params.model, data_dir=ds_dict, dataset_type=params.dataset_type, work_dir=params.output_dir, batch_bins=params.batch_bins, max_epoch=params.max_epoch, lr=params.lr)
    trainer = build_trainer(Trainers.speech_asr_trainer, default_args=kwargs)
    trainer.train()
if __name__ == '__main__':
    from funasr.utils.modelscope_param import modelscope_args
    params = modelscope_args(model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
    params.output_dir = './checkpoint'
    params.data_path = 'speech_asr_aishell1_trainsets'
    params.dataset_type = 'small'
    params.batch_bins = 2000
    params.max_epoch = 50
    params.lr = 5e-05
    params.download_mode = DownloadMode.FORCE_REDOWNLOAD
    modelscope_finetune(params)