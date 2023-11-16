from typing import List, Optional, Dict, Any, Union, Callable, Tuple
import logging
import multiprocessing
from pathlib import Path
from collections import defaultdict
import os
import tempfile
from time import perf_counter
from huggingface_hub import create_repo, HfFolder, Repository
from haystack.errors import HaystackError
from haystack.schema import Document, Answer, Span
from haystack.document_stores.base import BaseDocumentStore
from haystack.nodes.reader.base import BaseReader
from haystack.utils import get_batches_from_generator
from haystack.utils.early_stopping import EarlyStopping
from haystack.telemetry import send_event
from haystack.lazy_imports import LazyImport
logger = logging.getLogger(__name__)
with LazyImport(message="Run 'pip install farm-haystack[inference]'") as torch_and_transformers_import:
    import torch
    from haystack.modeling.data_handler.data_silo import DataSilo, DistillationDataSilo
    from haystack.modeling.data_handler.processor import SquadProcessor, Processor
    from haystack.modeling.data_handler.dataloader import NamedDataLoader
    from haystack.modeling.data_handler.inputs import QAInput, Question
    from haystack.modeling.infer import QAInferencer
    from haystack.modeling.model.optimization import initialize_optimizer
    from haystack.modeling.model.predictions import QAPred, QACandidate
    from haystack.modeling.model.adaptive_model import AdaptiveModel
    from haystack.modeling.training import Trainer, DistillationTrainer, TinyBERTDistillationTrainer
    from haystack.modeling.evaluation import Evaluator
    from haystack.modeling.utils import set_all_seeds, initialize_device_settings

class FARMReader(BaseReader):
    """
    Transformer based model for extractive Question Answering using the FARM framework (https://github.com/deepset-ai/FARM).
    While the underlying model can vary (BERT, Roberta, DistilBERT, ...), the interface remains the same.

    With a FARMReader, you can:

     - directly get predictions via predict()
     - fine-tune the model on QA data via train()
    """

    def __init__(self, model_name_or_path: str, model_version: Optional[str]=None, context_window_size: int=150, batch_size: int=50, use_gpu: bool=True, devices: Optional[List[Union[str, 'torch.device']]]=None, no_ans_boost: float=0.0, return_no_answer: bool=False, top_k: int=10, top_k_per_candidate: int=3, top_k_per_sample: int=1, num_processes: Optional[int]=None, max_seq_len: int=256, doc_stride: int=128, progress_bar: bool=True, duplicate_filtering: int=0, use_confidence_scores: bool=True, confidence_threshold: Optional[float]=None, proxies: Optional[Dict[str, str]]=None, local_files_only=False, force_download=False, use_auth_token: Optional[Union[str, bool]]=None, max_query_length: int=64, preprocessing_batch_size: Optional[int]=None):
        if False:
            return 10
        '\n        :param model_name_or_path: Directory of a saved model or the name of a public model e.g. \'bert-base-cased\',\n        \'deepset/bert-base-cased-squad2\', \'deepset/bert-base-cased-squad2\', \'distilbert-base-uncased-distilled-squad\'.\n        See https://huggingface.co/models for full list of available models.\n        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.\n        :param context_window_size: The size, in characters, of the window around the answer span that is used when\n                                    displaying the context around the answer.\n        :param batch_size: Number of samples the model receives in one batch for inference.\n                           Memory consumption is much lower in inference mode. Recommendation: Increase the batch size\n                           to a value so only a single batch is used.\n        :param use_gpu: Whether to use GPUs or the CPU. Falls back on CPU if no GPU is available.\n        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.\n                        A list containing torch device objects and/or strings is supported (For example\n                        [torch.device(\'cuda:0\'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices\n                        parameter is not used and a single cpu device is used for inference.\n        :param no_ans_boost: How much the no_answer logit is boosted/increased.\n        If set to 0 (default), the no_answer logit is not changed.\n        If a negative number, there is a lower chance of "no_answer" being predicted.\n        If a positive number, there is an increased chance of "no_answer"\n        :param return_no_answer: Whether to include no_answer predictions in the results.\n        :param top_k: The maximum number of answers to return\n        :param top_k_per_candidate: How many answers to extract for each candidate doc that is coming from the retriever (might be a long text).\n        Note that this is not the number of "final answers" you will receive\n        (see `top_k` in FARMReader.predict() or Finder.get_answers() for that)\n        and that FARM includes no_answer in the sorted list of predictions.\n        :param top_k_per_sample: How many answers to extract from each small text passage that the model can process at once\n        (one "candidate doc" is usually split into many smaller "passages").\n        You usually want a very small value here, as it slows down inference\n        and you don\'t gain much of quality by having multiple answers from one passage.\n        Note that this is not the number of "final answers" you will receive\n        (see `top_k` in FARMReader.predict() or Finder.get_answers() for that)\n        and that FARM includes no_answer in the sorted list of predictions.\n        :param num_processes: The number of processes for `multiprocessing.Pool`. Set to value of 0 to disable\n                              multiprocessing. Set to None to let Inferencer determine optimum number. If you\n                              want to debug the Language Model, you might need to disable multiprocessing!\n        :param max_seq_len: Max sequence length of one input text for the model\n        :param doc_stride: Length of striding window for splitting long texts (used if ``len(text) > max_seq_len``)\n        :param progress_bar: Whether to show a tqdm progress bar or not.\n                             Can be helpful to disable in production deployments to keep the logs clean.\n        :param duplicate_filtering: Answers are filtered based on their position. Both start and end position of the answers are considered.\n                                    The higher the value, answers that are more apart are filtered out. 0 corresponds to exact duplicates. -1 turns off duplicate removal.\n        :param use_confidence_scores: Determines the type of score that is used for ranking a predicted answer.\n                                      `True` => a scaled confidence / relevance score between [0, 1].\n                                      This score can also be further calibrated on your dataset via self.eval()\n                                      (see https://docs.haystack.deepset.ai/docs/reader#confidence-scores).\n                                      `False` => an unscaled, raw score [-inf, +inf] which is the sum of start and end logit\n                                      from the model for the predicted span.\n                                      Using confidence scores can change the ranking of no_answer compared to using the\n                                      unscaled raw scores.\n        :param confidence_threshold: Filters out predictions below confidence_threshold. Value should be between 0 and 1. Disabled by default.\n        :param proxies: Dict of proxy servers to use for downloading external models. Example: {\'http\': \'some.proxy:1234\', \'http://hostname\': \'my.proxy:3111\'}\n        :param local_files_only: Whether to force checking for local files only (and forbid downloads)\n        :param force_download: Whether to force a (re-)download even if the model exists locally in the cache.\n        :param use_auth_token: The API token used to download private models from Huggingface.\n                               If this parameter is set to `True`, then the token generated when running\n                               `transformers-cli login` (stored in ~/.huggingface) will be used.\n                               Additional information can be found here\n                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained\n        :param max_query_length: Maximum length of the question in number of tokens.\n        :param preprocessing_batch_size: Number of query-document pairs to be preprocessed (= tokenized, put into\n                                         tensors, etc.) at once. If `None` (default), all query-document pairs are\n                                         preprocessed at once.\n        '
        torch_and_transformers_import.check()
        super().__init__()
        (self.devices, self.n_gpu) = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=True)
        self.return_no_answers = return_no_answer
        self.top_k = top_k
        self.top_k_per_candidate = top_k_per_candidate
        self.inferencer = QAInferencer.load(model_name_or_path, batch_size=batch_size, gpu=self.n_gpu > 0, task_type='question_answering', max_seq_len=max_seq_len, doc_stride=doc_stride, num_processes=num_processes, revision=model_version, disable_tqdm=not progress_bar, strict=False, proxies=proxies, local_files_only=local_files_only, force_download=force_download, devices=self.devices, use_auth_token=use_auth_token, max_query_length=max_query_length)
        self.inferencer.model.prediction_heads[0].context_window_size = context_window_size
        self.inferencer.model.prediction_heads[0].no_ans_boost = no_ans_boost
        self.inferencer.model.prediction_heads[0].n_best = top_k_per_candidate + 1
        self.inferencer.model.prediction_heads[0].n_best_per_sample = top_k_per_sample
        self.inferencer.model.prediction_heads[0].duplicate_filtering = duplicate_filtering
        self.inferencer.model.prediction_heads[0].use_confidence_scores_for_ranking = use_confidence_scores
        self.max_seq_len = max_seq_len
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.progress_bar = progress_bar
        self.use_confidence_scores = use_confidence_scores
        self.confidence_threshold = confidence_threshold
        self.model_name_or_path = model_name_or_path
        self.preprocessing_batch_size = preprocessing_batch_size

    def _training_procedure(self, data_dir: str, train_filename: str, dev_filename: Optional[str]=None, test_filename: Optional[str]=None, use_gpu: Optional[bool]=None, devices: Optional[List['torch.device']]=None, batch_size: int=10, n_epochs: int=2, learning_rate: float=1e-05, max_seq_len: Optional[int]=None, warmup_proportion: float=0.2, dev_split: float=0, evaluate_every: int=300, save_dir: Optional[str]=None, num_processes: Optional[int]=None, use_amp: bool=False, checkpoint_root_dir: Path=Path('model_checkpoints'), checkpoint_every: Optional[int]=None, checkpoints_to_keep: int=3, teacher_model: Optional['FARMReader']=None, teacher_batch_size: Optional[int]=None, caching: bool=False, cache_path: Path=Path('cache/data_silo'), distillation_loss_weight: float=0.5, distillation_loss: Union[str, Callable[['torch.Tensor', 'torch.Tensor'], 'torch.Tensor']]='kl_div', temperature: float=1.0, tinybert: bool=False, processor: Optional['Processor']=None, grad_acc_steps: int=1, early_stopping: Optional['EarlyStopping']=None, distributed: bool=False, doc_stride: Optional[int]=None, max_query_length: Optional[int]=None):
        if False:
            while True:
                i = 10
        if devices is None:
            devices = []
        if dev_filename:
            dev_split = 0
        if num_processes is None:
            num_processes = multiprocessing.cpu_count() - 1 or 1
        set_all_seeds(seed=42)
        if devices is None:
            devices = self.devices
        if max_seq_len is None:
            max_seq_len = self.max_seq_len
        if doc_stride is None:
            doc_stride = self.doc_stride
        if max_query_length is None:
            max_query_length = self.max_query_length
        (devices, n_gpu) = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        if not save_dir:
            save_dir = f'./saved_models/{self.inferencer.model.language_model.name}'
            if tinybert:
                save_dir += '_tinybert_stage_1'
        label_list = ['start_token', 'end_token']
        metric = 'squad'
        if processor is None:
            processor = SquadProcessor(tokenizer=self.inferencer.processor.tokenizer, max_seq_len=max_seq_len, max_query_length=max_query_length, doc_stride=doc_stride, label_list=label_list, metric=metric, train_filename=train_filename, dev_filename=dev_filename, dev_split=dev_split, test_filename=test_filename, data_dir=Path(data_dir))
        data_silo: DataSilo
        if teacher_model and (not tinybert):
            data_silo = DistillationDataSilo(teacher_model, teacher_batch_size or batch_size, device=devices[0], processor=processor, batch_size=batch_size, distributed=distributed, max_processes=num_processes, caching=caching, cache_path=cache_path)
        else:
            data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=distributed, max_processes=num_processes, caching=caching, cache_path=cache_path)
        (model, optimizer, lr_schedule) = initialize_optimizer(model=self.inferencer.model, learning_rate=learning_rate, schedule_opts={'name': 'LinearWarmup', 'warmup_proportion': warmup_proportion}, n_batches=len(data_silo.loaders['train']), n_epochs=n_epochs, device=devices[0], grad_acc_steps=grad_acc_steps, distributed=distributed)
        if tinybert:
            if not teacher_model:
                raise ValueError('TinyBERT distillation requires a teacher model.')
            trainer = TinyBERTDistillationTrainer.create_or_load_checkpoint(model=model, teacher_model=teacher_model.inferencer.model, optimizer=optimizer, data_silo=data_silo, epochs=n_epochs, n_gpu=n_gpu, lr_schedule=lr_schedule, evaluate_every=evaluate_every, device=devices[0], use_amp=use_amp, disable_tqdm=not self.progress_bar, checkpoint_root_dir=Path(checkpoint_root_dir), checkpoint_every=checkpoint_every, checkpoints_to_keep=checkpoints_to_keep, grad_acc_steps=grad_acc_steps, early_stopping=early_stopping)
        elif teacher_model:
            trainer = DistillationTrainer.create_or_load_checkpoint(model=model, optimizer=optimizer, data_silo=data_silo, epochs=n_epochs, n_gpu=n_gpu, lr_schedule=lr_schedule, evaluate_every=evaluate_every, device=devices[0], use_amp=use_amp, disable_tqdm=not self.progress_bar, checkpoint_root_dir=Path(checkpoint_root_dir), checkpoint_every=checkpoint_every, checkpoints_to_keep=checkpoints_to_keep, distillation_loss=distillation_loss, distillation_loss_weight=distillation_loss_weight, temperature=temperature, grad_acc_steps=grad_acc_steps, early_stopping=early_stopping)
        else:
            trainer = Trainer.create_or_load_checkpoint(model=model, optimizer=optimizer, data_silo=data_silo, epochs=n_epochs, n_gpu=n_gpu, lr_schedule=lr_schedule, evaluate_every=evaluate_every, device=devices[0], use_amp=use_amp, disable_tqdm=not self.progress_bar, checkpoint_root_dir=Path(checkpoint_root_dir), checkpoint_every=checkpoint_every, checkpoints_to_keep=checkpoints_to_keep, grad_acc_steps=grad_acc_steps, early_stopping=early_stopping)
        self.inferencer.model = trainer.train()
        self.save(Path(save_dir))

    def train(self, data_dir: str, train_filename: str, dev_filename: Optional[str]=None, test_filename: Optional[str]=None, use_gpu: Optional[bool]=None, devices: Optional[List['torch.device']]=None, batch_size: int=10, n_epochs: int=2, learning_rate: float=1e-05, max_seq_len: Optional[int]=None, warmup_proportion: float=0.2, dev_split: float=0, evaluate_every: int=300, save_dir: Optional[str]=None, num_processes: Optional[int]=None, use_amp: bool=False, checkpoint_root_dir: Path=Path('model_checkpoints'), checkpoint_every: Optional[int]=None, checkpoints_to_keep: int=3, caching: bool=False, cache_path: Path=Path('cache/data_silo'), grad_acc_steps: int=1, early_stopping: Optional[EarlyStopping]=None, max_query_length: Optional[int]=None):
        if False:
            i = 10
            return i + 15
        '\n        Fine-tune a model on a QA dataset. Options:\n        - Take a plain language model (e.g. `bert-base-cased`) and train it for QA (e.g. on SQuAD data)\n        - Take a QA model (e.g. `deepset/bert-base-cased-squad2`) and fine-tune it for your domain (e.g. using your labels collected via the haystack annotation tool)\n\n        Checkpoints can be stored via setting `checkpoint_every` to a custom number of steps.\n        If any checkpoints are stored, a subsequent run of train() will resume training from the latest available checkpoint.\n\n        Note that when performing training with this function, long documents are split into chunks.\n        If a chunk doesn\'t contain the answer to the question, it is treated as a no-answer sample.\n\n        :param data_dir: Path to directory containing your training data in SQuAD style\n        :param train_filename: Filename of training data\n        :param dev_filename: Filename of dev / eval data\n        :param test_filename: Filename of test data\n        :param dev_split: Instead of specifying a dev_filename, you can also specify a ratio (e.g. 0.1) here\n                          that gets split off from training data for eval.\n        :param use_gpu: Whether to use GPU (if available)\n        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.\n                        A list containing torch device objects and/or strings is supported (For example\n                        [torch.device(\'cuda:0\'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices\n                        parameter is not used and a single cpu device is used for inference.\n        :param batch_size: Number of samples the model receives in one batch for training\n        :param n_epochs: Number of iterations on the whole training data set\n        :param learning_rate: Learning rate of the optimizer\n        :param max_seq_len: Maximum text length (in tokens). Everything longer gets cut down.\n        :param warmup_proportion: Proportion of training steps until maximum learning rate is reached.\n                                  Until that point LR is increasing linearly. After that it\'s decreasing again linearly.\n                                  Options for different schedules are available in FARM.\n        :param evaluate_every: Evaluate the model every X steps on the hold-out eval dataset.\n                               Note that the evaluation report is logged at evaluation level INFO while Haystack\'s default is WARNING.\n        :param save_dir: Path to store the final model\n        :param num_processes: The number of processes for `multiprocessing.Pool` during preprocessing.\n                              Set to value of 1 to disable multiprocessing. When set to 1, you cannot split away a dev set from train set.\n                              Set to None to use all CPU cores minus one.\n        :param use_amp: Whether to use automatic mixed precision (AMP) natively implemented in PyTorch to improve\n                        training speed and reduce GPU memory usage.\n                        For more information, see (Haystack Optimization)[https://haystack.deepset.ai/guides/optimization]\n                        and (Automatic Mixed Precision Package - Torch.amp)[https://pytorch.org/docs/stable/amp.html].\n        :param checkpoint_root_dir: The Path of a directory where all train checkpoints are saved. For each individual\n               checkpoint, a subdirectory with the name epoch_{epoch_num}_step_{step_num} is created.\n        :param checkpoint_every: Save a train checkpoint after this many steps of training.\n        :param checkpoints_to_keep: The maximum number of train checkpoints to save.\n        :param caching: Whether or not to use caching for the preprocessed dataset.\n        :param cache_path: The Path to cache the preprocessed dataset.\n        :param grad_acc_steps: The number of steps to accumulate gradients for before performing a backward pass.\n        :param early_stopping: An initialized EarlyStopping object to control early stopping and saving of the best models.\n        :param max_query_length: Maximum length of the question in number of tokens.\n        :return: None\n        '
        send_event(event_name='Training', event_properties={'class': self.__class__.__name__, 'function_name': 'train'})
        return self._training_procedure(data_dir=data_dir, train_filename=train_filename, dev_filename=dev_filename, test_filename=test_filename, use_gpu=use_gpu, devices=devices, batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate, max_seq_len=max_seq_len, warmup_proportion=warmup_proportion, dev_split=dev_split, evaluate_every=evaluate_every, save_dir=save_dir, num_processes=num_processes, use_amp=use_amp, checkpoint_root_dir=checkpoint_root_dir, checkpoint_every=checkpoint_every, checkpoints_to_keep=checkpoints_to_keep, caching=caching, cache_path=cache_path, grad_acc_steps=grad_acc_steps, early_stopping=early_stopping, max_query_length=max_query_length, distributed=False)

    def distil_prediction_layer_from(self, teacher_model: 'FARMReader', data_dir: str, train_filename: str, dev_filename: Optional[str]=None, test_filename: Optional[str]=None, use_gpu: Optional[bool]=None, devices: Optional[List['torch.device']]=None, batch_size: int=10, teacher_batch_size: Optional[int]=None, n_epochs: int=2, learning_rate: float=3e-05, max_seq_len: Optional[int]=None, warmup_proportion: float=0.2, dev_split: float=0, evaluate_every: int=300, save_dir: Optional[str]=None, num_processes: Optional[int]=None, use_amp: bool=False, checkpoint_root_dir: Path=Path('model_checkpoints'), checkpoint_every: Optional[int]=None, checkpoints_to_keep: int=3, caching: bool=False, cache_path: Path=Path('cache/data_silo'), distillation_loss_weight: float=0.5, distillation_loss: Union[str, Callable[['torch.Tensor', 'torch.Tensor'], 'torch.Tensor']]='kl_div', temperature: float=1.0, processor: Optional['Processor']=None, grad_acc_steps: int=1, early_stopping: Optional['EarlyStopping']=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fine-tune a model on a QA dataset using logit-based distillation. You need to provide a teacher model that is already finetuned on the dataset\n        and a student model that will be trained using the teacher\'s logits. The idea of this is to increase the accuracy of a lightweight student model.\n        using a more complex teacher.\n        Originally proposed in: https://arxiv.org/pdf/1503.02531.pdf\n        This can also be considered as the second stage of distillation finetuning as described in the TinyBERT paper:\n        https://arxiv.org/pdf/1909.10351.pdf\n        **Example**\n        ```python\n        student = FARMReader(model_name_or_path="prajjwal1/bert-medium")\n        teacher = FARMReader(model_name_or_path="deepset/bert-large-uncased-whole-word-masking-squad2")\n        student.distil_prediction_layer_from(teacher, data_dir="squad2", train_filename="train.json", test_filename="dev.json",\n                            learning_rate=3e-5, distillation_loss_weight=1.0, temperature=5)\n        ```\n\n        Checkpoints can be stored via setting `checkpoint_every` to a custom number of steps.\n        If any checkpoints are stored, a subsequent run of train() will resume training from the latest available checkpoint.\n\n        :param teacher_model: Model whose logits will be used to improve accuracy\n        :param data_dir: Path to directory containing your training data in SQuAD style\n        :param train_filename: Filename of training data\n        :param dev_filename: Filename of dev / eval data\n        :param test_filename: Filename of test data\n        :param dev_split: Instead of specifying a dev_filename, you can also specify a ratio (e.g. 0.1) here\n                          that gets split off from training data for eval.\n        :param use_gpu: Whether to use GPU (if available)\n        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.\n                        A list containing torch device objects and/or strings is supported (For example\n                        [torch.device(\'cuda:0\'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices\n                        parameter is not used and a single cpu device is used for inference.\n        :param batch_size: Number of samples the student model receives in one batch for training\n        :param teacher_batch_size: Number of samples the teacher model receives in one batch for distillation\n        :param n_epochs: Number of iterations on the whole training data set\n        :param learning_rate: Learning rate of the optimizer\n        :param max_seq_len: Maximum text length (in tokens). Everything longer gets cut down.\n        :param warmup_proportion: Proportion of training steps until maximum learning rate is reached.\n                                  Until that point LR is increasing linearly. After that it\'s decreasing again linearly.\n                                  Options for different schedules are available in FARM.\n        :param evaluate_every: Evaluate the model every X steps on the hold-out eval dataset\n        :param save_dir: Path to store the final model\n        :param num_processes: The number of processes for `multiprocessing.Pool` during preprocessing.\n                              Set to value of 1 to disable multiprocessing. When set to 1, you cannot split away a dev set from train set.\n                              Set to None to use all CPU cores minus one.\n        :param use_amp: Whether to use automatic mixed precision (AMP) natively implemented in PyTorch to improve\n                        training speed and reduce GPU memory usage.\n                        For more information, see (Haystack Optimization)[https://haystack.deepset.ai/guides/optimization]\n                        and (Automatic Mixed Precision Package - Torch.amp)[https://pytorch.org/docs/stable/amp.html].\n        :param checkpoint_root_dir: the Path of directory where all train checkpoints are saved. For each individual\n               checkpoint, a subdirectory with the name epoch_{epoch_num}_step_{step_num} is created.\n        :param checkpoint_every: save a train checkpoint after this many steps of training.\n        :param checkpoints_to_keep: maximum number of train checkpoints to save.\n        :param caching: whether or not to use caching for preprocessed dataset and teacher logits\n        :param cache_path: Path to cache the preprocessed dataset and teacher logits\n        :param distillation_loss_weight: The weight of the distillation loss. A higher weight means the teacher outputs are more important.\n        :param distillation_loss: Specifies how teacher and model logits should be compared. Can either be a string ("mse" for mean squared error or "kl_div" for kl divergence loss) or a callable loss function (needs to have named parameters student_logits and teacher_logits)\n        :param temperature: The temperature for distillation. A higher temperature will result in less certainty of teacher outputs. A lower temperature means more certainty. A temperature of 1.0 does not change the certainty of the model.\n        :param processor: The processor to use for preprocessing. If None, the default SquadProcessor is used.\n        :param grad_acc_steps: The number of steps to accumulate gradients for before performing a backward pass.\n        :param early_stopping: An initialized EarlyStopping object to control early stopping and saving of the best models.\n        :return: None\n        '
        send_event(event_name='Training', event_properties={'class': self.__class__.__name__, 'function_name': 'distil_prediction_layer_from'})
        return self._training_procedure(data_dir=data_dir, train_filename=train_filename, dev_filename=dev_filename, test_filename=test_filename, use_gpu=use_gpu, devices=devices, batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate, max_seq_len=max_seq_len, warmup_proportion=warmup_proportion, dev_split=dev_split, evaluate_every=evaluate_every, save_dir=save_dir, num_processes=num_processes, use_amp=use_amp, checkpoint_root_dir=checkpoint_root_dir, checkpoint_every=checkpoint_every, checkpoints_to_keep=checkpoints_to_keep, teacher_model=teacher_model, teacher_batch_size=teacher_batch_size, caching=caching, cache_path=cache_path, distillation_loss_weight=distillation_loss_weight, distillation_loss=distillation_loss, temperature=temperature, processor=processor, grad_acc_steps=grad_acc_steps, early_stopping=early_stopping, distributed=False)

    def distil_intermediate_layers_from(self, teacher_model: 'FARMReader', data_dir: str, train_filename: str, dev_filename: Optional[str]=None, test_filename: Optional[str]=None, use_gpu: Optional[bool]=None, devices: Optional[List['torch.device']]=None, batch_size: int=10, teacher_batch_size: Optional[int]=None, n_epochs: int=5, learning_rate: float=5e-05, max_seq_len: Optional[int]=None, warmup_proportion: float=0.2, dev_split: float=0, evaluate_every: int=300, save_dir: Optional[str]=None, num_processes: Optional[int]=None, use_amp: bool=False, checkpoint_root_dir: Path=Path('model_checkpoints'), checkpoint_every: Optional[int]=None, checkpoints_to_keep: int=3, caching: bool=False, cache_path: Path=Path('cache/data_silo'), distillation_loss: Union[str, Callable[['torch.Tensor', 'torch.Tensor'], 'torch.Tensor']]='mse', distillation_loss_weight: float=0.5, temperature: float=1.0, processor: Optional['Processor']=None, grad_acc_steps: int=1, early_stopping: Optional['EarlyStopping']=None):
        if False:
            while True:
                i = 10
        '\n        The first stage of distillation finetuning as described in the TinyBERT paper:\n        https://arxiv.org/pdf/1909.10351.pdf\n        **Example**\n        ```python\n        student = FARMReader(model_name_or_path="prajjwal1/bert-medium")\n        teacher = FARMReader(model_name_or_path="huawei-noah/TinyBERT_General_6L_768D")\n        student.distil_intermediate_layers_from(teacher, data_dir="squad2", train_filename="train.json", test_filename="dev.json",\n                            learning_rate=3e-5, distillation_loss_weight=1.0, temperature=5)\n        ```\n\n        Checkpoints can be stored via setting `checkpoint_every` to a custom number of steps.\n        If any checkpoints are stored, a subsequent run of train() will resume training from the latest available checkpoint.\n\n        :param teacher_model: Model whose logits will be used to improve accuracy\n        :param data_dir: Path to directory containing your training data in SQuAD style\n        :param train_filename: Filename of training data. To best follow the original paper, this should be an augmented version of the training data created using the augment_squad.py script\n        :param dev_filename: Filename of dev / eval data\n        :param test_filename: Filename of test data\n        :param dev_split: Instead of specifying a dev_filename, you can also specify a ratio (e.g. 0.1) here\n                          that gets split off from training data for eval.\n        :param use_gpu: Whether to use GPU (if available)\n        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.\n                        A list containing torch device objects and/or strings is supported (For example\n                        [torch.device(\'cuda:0\'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices\n                        parameter is not used and a single cpu device is used for inference.\n        :param batch_size: Number of samples the student model receives in one batch for training\n        :param teacher_batch_size: Number of samples the teacher model receives in one batch for distillation.\n        :param n_epochs: Number of iterations on the whole training data set\n        :param learning_rate: Learning rate of the optimizer\n        :param max_seq_len: Maximum text length (in tokens). Everything longer gets cut down.\n        :param warmup_proportion: Proportion of training steps until maximum learning rate is reached.\n                                  Until that point LR is increasing linearly. After that it\'s decreasing again linearly.\n                                  Options for different schedules are available in FARM.\n        :param evaluate_every: Evaluate the model every X steps on the hold-out eval dataset\n        :param save_dir: Path to store the final model\n        :param num_processes: The number of processes for `multiprocessing.Pool` during preprocessing.\n                              Set to value of 1 to disable multiprocessing. When set to 1, you cannot split away a dev set from train set.\n                              Set to None to use all CPU cores minus one.\n        :param use_amp: Whether to use automatic mixed precision (AMP) natively implemented in PyTorch to improve\n                        training speed and reduce GPU memory usage.\n                        For more information, see (Haystack Optimization)[https://haystack.deepset.ai/guides/optimization]\n                        and (Automatic Mixed Precision Package - Torch.amp)[https://pytorch.org/docs/stable/amp.html].\n        :param checkpoint_root_dir: the Path of directory where all train checkpoints are saved. For each individual\n               checkpoint, a subdirectory with the name epoch_{epoch_num}_step_{step_num} is created.\n        :param checkpoint_every: save a train checkpoint after this many steps of training.\n        :param checkpoints_to_keep: maximum number of train checkpoints to save.\n        :param caching: whether or not to use caching for preprocessed dataset and teacher logits\n        :param cache_path: Path to cache the preprocessed dataset and teacher logits\n        :param distillation_loss: Specifies how teacher and model logits should be compared. Can either be a string ("mse" for mean squared error or "kl_div" for kl divergence loss) or a callable loss function (needs to have named parameters student_logits and teacher_logits)\n        :param distillation_loss_weight: The weight of the distillation loss. A higher weight means the teacher outputs are more important.\n        :param temperature: The temperature for distillation. A higher temperature will result in less certainty of teacher outputs. A lower temperature means more certainty. A temperature of 1.0 does not change the certainty of the model.\n        :param processor: The processor to use for preprocessing. If None, the default SquadProcessor is used.\n        :param grad_acc_steps: The number of steps to accumulate gradients for before performing a backward pass.\n        :param early_stopping: An initialized EarlyStopping object to control early stopping and saving of the best models.\n        :return: None\n        '
        send_event(event_name='Training', event_properties={'class': self.__class__.__name__, 'function_name': 'distil_intermediate_layers_from'})
        return self._training_procedure(data_dir=data_dir, train_filename=train_filename, dev_filename=dev_filename, test_filename=test_filename, use_gpu=use_gpu, devices=devices, batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate, max_seq_len=max_seq_len, warmup_proportion=warmup_proportion, dev_split=dev_split, evaluate_every=evaluate_every, save_dir=save_dir, num_processes=num_processes, use_amp=use_amp, checkpoint_root_dir=checkpoint_root_dir, checkpoint_every=checkpoint_every, checkpoints_to_keep=checkpoints_to_keep, teacher_model=teacher_model, teacher_batch_size=teacher_batch_size, caching=caching, cache_path=cache_path, distillation_loss=distillation_loss, distillation_loss_weight=distillation_loss_weight, temperature=temperature, tinybert=True, processor=processor, grad_acc_steps=grad_acc_steps, early_stopping=early_stopping, distributed=False)

    def update_parameters(self, context_window_size: Optional[int]=None, no_ans_boost: Optional[float]=None, return_no_answer: Optional[bool]=None, max_seq_len: Optional[int]=None, doc_stride: Optional[int]=None):
        if False:
            i = 10
            return i + 15
        '\n        Hot update parameters of a loaded Reader. It may not to be safe when processing concurrent requests.\n        '
        if no_ans_boost is not None:
            self.inferencer.model.prediction_heads[0].no_ans_boost = no_ans_boost
        if return_no_answer is not None:
            self.return_no_answers = return_no_answer
        if doc_stride is not None:
            self.inferencer.processor.doc_stride = doc_stride
        if context_window_size is not None:
            self.inferencer.model.prediction_heads[0].context_window_size = context_window_size
        if max_seq_len is not None:
            self.inferencer.processor.max_seq_len = max_seq_len
            self.max_seq_len = max_seq_len

    def save(self, directory: Path):
        if False:
            for i in range(10):
                print('nop')
        '\n        Saves the Reader model so that it can be reused at a later point in time.\n\n        :param directory: Directory where the Reader model should be saved\n        '
        logger.info('Saving reader model to %s', directory)
        self.inferencer.model.save(directory)
        self.inferencer.processor.save(directory)

    def save_to_remote(self, repo_id: str, private: bool=False, commit_message: str='Add new model to Hugging Face.'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Saves the Reader model to Hugging Face Model Hub with the given model_name. For this to work:\n        - Be logged in to Hugging Face on your machine via transformers-cli\n        - Have git lfs installed (https://packagecloud.io/github/git-lfs/install), you can test it by git lfs --version\n\n        :param repo_id: A namespace (user or an organization) and a repo name separated by a '/' of the model you want to save to Hugging Face\n        :param private: Set to true to make the model repository private\n        :param commit_message: Commit message while saving to Hugging Face\n        "
        token = HfFolder.get_token()
        if token is None:
            raise ValueError('To save this reader model to Hugging Face, make sure you login to the hub on this computer by typing `transformers-cli login`.')
        repo_url = create_repo(token=token, repo_id=repo_id, private=private, repo_type=None, exist_ok=True)
        transformer_models = self.inferencer.model.convert_to_transformers()
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = Repository(tmp_dir, clone_from=repo_url)
            self.inferencer.processor.tokenizer.save_pretrained(tmp_dir)
            transformer_models[0].save_pretrained(tmp_dir)
            large_files = []
            for (root, _, files) in os.walk(tmp_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, tmp_dir)
                    if os.path.getsize(file_path) > 5 * 1024 * 1024:
                        large_files.append(rel_path)
            if len(large_files) > 0:
                logger.info('Track files with git lfs: %s', ', '.join(large_files))
                repo.lfs_track(large_files)
            logger.info('Push model to the hub. This might take a while')
            commit_url = repo.push_to_hub(commit_message=commit_message)
        return commit_url

    def predict_batch(self, queries: List[str], documents: Union[List[Document], List[List[Document]]], top_k: Optional[int]=None, batch_size: Optional[int]=None):
        if False:
            return 10
        '\n        Use loaded QA model to find answers for the queries in the Documents.\n\n        - If you provide a list containing a single query...\n\n            - ... and a single list of Documents, the query will be applied to each Document individually.\n            - ... and a list of lists of Documents, the query will be applied to each list of Documents and the Answers\n              will be aggregated per Document list.\n\n        - If you provide a list of multiple queries...\n\n            - ... and a single list of Documents, each query will be applied to each Document individually.\n            - ... and a list of lists of Documents, each query will be applied to its corresponding list of Documents\n              and the Answers will be aggregated per query-Document pair.\n\n        :param queries: Single query or list of queries.\n        :param documents: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.\n                          Can be a single list of Documents or a list of lists of Documents.\n        :param top_k: Number of returned answers per query.\n        :param batch_size: Number of query-document pairs to be processed at a time.\n        '
        if top_k is None:
            top_k = self.top_k
        (inputs, number_of_docs, single_doc_list) = self._preprocess_batch_queries_and_docs(queries=queries, documents=documents)
        if batch_size is not None:
            self.inferencer.batch_size = batch_size
        predictions = []
        for input_batch in get_batches_from_generator(inputs, self.preprocessing_batch_size):
            cur_predictions = self.inferencer.inference_from_objects(objects=input_batch, return_json=False, multiprocessing_chunksize=10)
            predictions.extend(cur_predictions)
        grouped_predictions = []
        left_idx = 0
        for number in number_of_docs:
            right_idx = left_idx + number
            grouped_predictions.append(predictions[left_idx:right_idx])
            left_idx = right_idx
        results: Dict = {'queries': queries, 'answers': [], 'no_ans_gaps': []}
        for group in grouped_predictions:
            (answers, max_no_ans_gap) = self._extract_answers_of_predictions(group, top_k)
            results['answers'].append(answers)
            results['no_ans_gaps'].append(max_no_ans_gap)
        if single_doc_list and len(queries) > 1:
            answers_per_query = int(len(results['answers']) / len(queries))
            answers = []
            for i in range(0, len(results['answers']), answers_per_query):
                answer_group = results['answers'][i:i + answers_per_query]
                answers.append(answer_group)
            results['answers'] = answers
        return results

    def predict(self, query: str, documents: List[Document], top_k: Optional[int]=None):
        if False:
            i = 10
            return i + 15
        '\n        Use loaded QA model to find answers for a query in the supplied list of Document.\n\n        Returns dictionaries containing answers sorted by (desc.) score.\n        Example:\n\n        ```python\n        {\n            \'query\': \'Who is the father of Arya Stark?\',\n            \'answers\':[Answer(\n                         \'answer\': \'Eddard,\',\n                         \'context\': "She travels with her father, Eddard, to King\'s Landing when he is",\n                         \'score\': 0.9787139466668613,\n                         \'offsets_in_document\': [Span(start=29, end=35],\n                         \'offsets_in_context\': [Span(start=347, end=353],\n                         \'document_id\': \'88d1ed769d003939d3a0d28034464ab2\'\n                         ),...\n                      ]\n        }\n         ```\n\n        :param query: Query string\n        :param documents: List of Document in which to search for the answer\n        :param top_k: The maximum number of answers to return\n        :return: Dict containing query and answers\n        '
        if top_k is None:
            top_k = self.top_k
        inputs = []
        for doc in documents:
            cur = QAInput(doc_text=doc.content, questions=Question(text=query, uid=doc.id))
            inputs.append(cur)
        predictions = []
        for input_batch in get_batches_from_generator(inputs, self.preprocessing_batch_size):
            cur_predictions = self.inferencer.inference_from_objects(objects=input_batch, return_json=False, multiprocessing_chunksize=1)
            predictions.extend(cur_predictions)
        predictions = self._deduplicate_predictions(predictions, documents)
        (answers, max_no_ans_gap) = self._extract_answers_of_predictions(predictions, top_k)
        result = {'query': query, 'no_ans_gap': max_no_ans_gap, 'answers': answers}
        return result

    def eval_on_file(self, data_dir: Union[Path, str], test_filename: str, device: Optional[Union[str, 'torch.device']]=None, calibrate_conf_scores: bool=False):
        if False:
            return 10
        '\n        Performs evaluation on a SQuAD-formatted file.\n        Returns a dict containing the following metrics:\n            - "EM": exact match score\n            - "f1": F1-Score\n            - "top_n_accuracy": Proportion of predicted answers that overlap with correct answer\n\n        :param data_dir: The directory in which the test set can be found\n        :param test_filename: The name of the file containing the test data in SQuAD format.\n        :param device: The device on which the tensors should be processed.\n               Choose from torch.device("cpu") and torch.device("cuda") (or simply "cpu" or "cuda")\n               or use the Reader\'s device by default.\n        :param calibrate_conf_scores: Whether to calibrate the temperature for scaling of the confidence scores.\n        '
        logger.warning("FARMReader.eval_on_file() uses a slightly different evaluation approach than `Pipeline.eval()`:\n- instead of giving you full control over which labels to use, this method always returns three types of metrics: combined (no suffix), text_answer ('_text_answer' suffix) and no_answer ('_no_answer' suffix) metrics.\n- instead of comparing predictions with labels on a string level, this method compares them on a token-ID level. This makes it unable to do any string normalization (e.g. normalize whitespaces) beforehand.\nHence, results might slightly differ from those of `Pipeline.eval()`\n.If you are just about starting to evaluate your model consider using `Pipeline.eval()` instead.")
        send_event(event_name='Evaluation', event_properties={'class': self.__class__.__name__, 'function_name': 'eval_on_file'})
        if device is None:
            device = self.devices[0]
        else:
            device = torch.device(device)
        eval_processor = SquadProcessor(tokenizer=self.inferencer.processor.tokenizer, max_seq_len=self.inferencer.processor.max_seq_len, label_list=self.inferencer.processor.tasks['question_answering']['label_list'], metric=self.inferencer.processor.tasks['question_answering']['metric'], train_filename=None, dev_filename=None, dev_split=0, test_filename=test_filename, data_dir=Path(data_dir))
        data_silo = DataSilo(processor=eval_processor, batch_size=self.inferencer.batch_size, distributed=False)
        data_loader = data_silo.get_data_loader('test')
        evaluator = Evaluator(data_loader=data_loader, tasks=eval_processor.tasks, device=device)
        eval_results = evaluator.eval(self.inferencer.model, calibrate_conf_scores=calibrate_conf_scores, use_confidence_scores_for_ranking=self.use_confidence_scores)
        results = {'EM': eval_results[0]['EM'] * 100, 'f1': eval_results[0]['f1'] * 100, 'top_n_accuracy': eval_results[0]['top_n_accuracy'] * 100, 'top_n': self.inferencer.model.prediction_heads[0].n_best, 'EM_text_answer': eval_results[0]['EM_text_answer'] * 100, 'f1_text_answer': eval_results[0]['f1_text_answer'] * 100, 'top_n_accuracy_text_answer': eval_results[0]['top_n_accuracy_text_answer'] * 100, 'top_n_EM_text_answer': eval_results[0]['top_n_EM_text_answer'] * 100, 'top_n_f1_text_answer': eval_results[0]['top_n_f1_text_answer'] * 100, 'Total_text_answer': eval_results[0]['Total_text_answer'], 'EM_no_answer': eval_results[0]['EM_no_answer'] * 100, 'f1_no_answer': eval_results[0]['f1_no_answer'] * 100, 'top_n_accuracy_no_answer': eval_results[0]['top_n_accuracy_no_answer'] * 100, 'Total_no_answer': eval_results[0]['Total_no_answer']}
        return results

    def eval(self, document_store: BaseDocumentStore, device: Optional[Union[str, 'torch.device']]=None, label_index: str='label', doc_index: str='eval_document', label_origin: str='gold-label', calibrate_conf_scores: bool=False):
        if False:
            while True:
                i = 10
        '\n        Performs evaluation on evaluation documents in the DocumentStore.\n        Returns a dict containing the following metrics:\n              - "EM": Proportion of exact matches of predicted answers with their corresponding correct answers\n              - "f1": Average overlap between predicted answers and their corresponding correct answers\n              - "top_n_accuracy": Proportion of predicted answers that overlap with correct answer\n\n        :param document_store: DocumentStore containing the evaluation documents\n        :param device: The device on which the tensors should be processed.\n                       Choose from torch.device("cpu") and torch.device("cuda") (or simply "cpu" or "cuda")\n                       or use the Reader\'s device by default.\n        :param label_index: Index/Table name where labeled questions are stored\n        :param doc_index: Index/Table name where documents that are used for evaluation are stored\n        :param label_origin: Field name where the gold labels are stored\n        :param calibrate_conf_scores: Whether to calibrate the temperature for scaling of the confidence scores.\n        '
        logger.warning("FARMReader.eval() uses a slightly different evaluation approach than `Pipeline.eval()`:\n- instead of giving you full control over which labels to use, this method always returns three types of metrics: combined (no suffix), text_answer ('_text_answer' suffix) and no_answer ('_no_answer' suffix) metrics.\n- instead of comparing predictions with labels on a string level, this method compares them on a token-ID level. This makes it unable to do any string normalization (e.g. normalize whitespaces) beforehand.\nHence, results might slightly differ from those of `Pipeline.eval()`\n.If you are just about starting to evaluate your model consider using `Pipeline.eval()` instead.")
        send_event(event_name='Evaluation', event_properties={'class': self.__class__.__name__, 'function_name': 'eval'})
        if device is None:
            device = self.devices[0]
        else:
            device = torch.device(device)
        if self.top_k_per_candidate != 4:
            logger.info("Performing Evaluation using top_k_per_candidate = %s \nand consequently, QuestionAnsweringPredictionHead.n_best = {self.top_k_per_candidate + 1}. \nThis deviates from FARM's default where QuestionAnsweringPredictionHead.n_best = 5", self.top_k_per_candidate)
        filters: Dict = {'origin': [label_origin]}
        labels = document_store.get_all_labels(index=label_index, filters=filters)
        aggregated_per_doc = defaultdict(list)
        for label in labels:
            if not label.document.id:
                logger.error('Label does not contain a document id')
                continue
            if label.document.content_type == 'table':
                logger.warning('Label with a table document is not compatible with the FARMReader. Skipping label with id %s.', label.id)
                continue
            aggregated_per_doc[label.document.id].append(label)
        d: Dict[str, Any] = {}
        all_doc_ids = [x.id for x in document_store.get_all_documents(doc_index)]
        for doc_id in all_doc_ids:
            doc = document_store.get_document_by_id(doc_id, index=doc_index)
            if not doc:
                logger.error("Document with the ID '%s' is not present in the document store.", doc_id)
                continue
            d[str(doc_id)] = {'context': doc.content}
            aggregated_per_question: Dict[tuple, Any] = defaultdict(list)
            if doc_id in aggregated_per_doc:
                for label in aggregated_per_doc[doc_id]:
                    aggregation_key = (doc_id, label.query)
                    if label.answer is None:
                        logger.error('Label.answer was None, but Answer object was expected: %s', label)
                        continue
                    if label.answer.offsets_in_document is None:
                        logger.error('Label.answer.offsets_in_document was None, but Span object was expected: %s ', label)
                        continue
                    if aggregation_key in aggregated_per_question.keys():
                        if label.no_answer:
                            continue
                        if len(aggregated_per_question[aggregation_key]['answers']) >= 6:
                            logger.warning('Answers in this sample are being dropped because it has more than 6 answers. (doc_id: %s, question: %s, label_id: %s)', doc_id, label.query, label.id)
                            continue
                        aggregated_per_question[aggregation_key]['answers'].append({'text': label.answer.answer, 'answer_start': label.answer.offsets_in_document[0].start})
                        aggregated_per_question[aggregation_key]['is_impossible'] = False
                    elif label.no_answer is True:
                        aggregated_per_question[aggregation_key] = {'id': str(hash(str(doc_id) + label.query)), 'question': label.query, 'answers': [], 'is_impossible': True}
                    else:
                        aggregated_per_question[aggregation_key] = {'id': str(hash(str(doc_id) + label.query)), 'question': label.query, 'answers': [{'text': label.answer.answer, 'answer_start': label.answer.offsets_in_document[0].start}], 'is_impossible': False}
            d[str(doc_id)]['qas'] = list(aggregated_per_question.values())
        farm_input = list(d.values())
        n_queries = len([y for x in farm_input for y in x['qas']])
        tic = perf_counter()
        indices = range(len(farm_input))
        (dataset, tensor_names, _) = self.inferencer.processor.dataset_from_dicts(farm_input, indices=indices)
        data_loader = NamedDataLoader(dataset=dataset, batch_size=self.inferencer.batch_size, tensor_names=tensor_names)
        evaluator = Evaluator(data_loader=data_loader, tasks=self.inferencer.processor.tasks, device=device)
        eval_results = evaluator.eval(self.inferencer.model, calibrate_conf_scores=calibrate_conf_scores, use_confidence_scores_for_ranking=self.use_confidence_scores)
        toc = perf_counter()
        reader_time = toc - tic
        results = {'EM': eval_results[0]['EM'] * 100, 'f1': eval_results[0]['f1'] * 100, 'top_n_accuracy': eval_results[0]['top_n_accuracy'] * 100, 'top_n': self.inferencer.model.prediction_heads[0].n_best, 'reader_time': reader_time, 'seconds_per_query': reader_time / n_queries, 'EM_text_answer': eval_results[0]['EM_text_answer'] * 100, 'f1_text_answer': eval_results[0]['f1_text_answer'] * 100, 'top_n_accuracy_text_answer': eval_results[0]['top_n_accuracy_text_answer'] * 100, 'top_n_EM_text_answer': eval_results[0]['top_n_EM_text_answer'] * 100, 'top_n_f1_text_answer': eval_results[0]['top_n_f1_text_answer'] * 100, 'Total_text_answer': eval_results[0]['Total_text_answer'], 'EM_no_answer': eval_results[0]['EM_no_answer'] * 100, 'f1_no_answer': eval_results[0]['f1_no_answer'] * 100, 'top_n_accuracy_no_answer': eval_results[0]['top_n_accuracy_no_answer'] * 100, 'Total_no_answer': eval_results[0]['Total_no_answer']}
        return results

    def _extract_answers_of_predictions(self, predictions: List['QAPred'], top_k: Optional[int]=None):
        if False:
            i = 10
            return i + 15
        answers: List[Answer] = []
        no_ans_gaps = []
        best_score_answer = 0
        for pred in predictions:
            answers_per_document = []
            no_ans_gaps.append(pred.no_answer_gap)
            for ans in pred.prediction:
                if self._check_no_answer(ans):
                    pass
                else:
                    cur = Answer(answer=ans.answer, type='extractive', score=ans.confidence if self.use_confidence_scores else ans.score, context=ans.context_window, document_ids=[pred.id], offsets_in_context=[Span(start=ans.offset_answer_start - ans.offset_context_window_start, end=ans.offset_answer_end - ans.offset_context_window_start)], offsets_in_document=[Span(start=ans.offset_answer_start, end=ans.offset_answer_end)])
                    answers_per_document.append(cur)
                    if ans.score > best_score_answer:
                        best_score_answer = ans.score
            answers += answers_per_document[:self.top_k_per_candidate]
        (no_ans_prediction, max_no_ans_gap) = self._calc_no_answer(no_ans_gaps, best_score_answer, self.use_confidence_scores)
        if self.return_no_answers:
            answers.append(no_ans_prediction)
        answers = sorted(answers, reverse=True)
        answers = answers[:top_k]
        if self.confidence_threshold is not None:
            answers = [ans for ans in answers if ans.score is not None and ans.score >= self.confidence_threshold]
        return (answers, max_no_ans_gap)

    def _preprocess_batch_queries_and_docs(self, queries: List[str], documents: Union[List[Document], List[List[Document]]]) -> Tuple[List['QAInput'], List[int], bool]:
        if False:
            while True:
                i = 10
        inputs = []
        number_of_docs = []
        single_doc_list = False
        if len(documents) > 0 and isinstance(documents[0], Document):
            single_doc_list = True
            for query in queries:
                for doc in documents:
                    number_of_docs.append(1)
                    if not isinstance(doc, Document):
                        raise HaystackError(f'doc was of type {type(doc)}, but expected a Document.')
                    cur = QAInput(doc_text=doc.content, questions=Question(text=query, uid=doc.id))
                    inputs.append(cur)
        elif len(documents) > 0 and isinstance(documents[0], list):
            single_doc_list = False
            if len(queries) == 1:
                queries = queries * len(documents)
            if len(queries) != len(documents):
                raise HaystackError('Number of queries must be equal to number of provided Document lists.')
            for (query, cur_docs) in zip(queries, documents):
                if not isinstance(cur_docs, list):
                    raise HaystackError(f'cur_docs was of type {type(cur_docs)}, but expected a list of Documents.')
                number_of_docs.append(len(cur_docs))
                for doc in cur_docs:
                    if not isinstance(doc, Document):
                        raise HaystackError(f'doc was of type {type(doc)}, but expected a Document.')
                    cur = QAInput(doc_text=doc.content, questions=Question(text=query, uid=doc.id))
                    inputs.append(cur)
        return (inputs, number_of_docs, single_doc_list)

    def _deduplicate_predictions(self, predictions: List['QAPred'], documents: List[Document]) -> List['QAPred']:
        if False:
            print('Hello World!')
        overlapping_docs = self._identify_overlapping_docs(documents)
        if not overlapping_docs:
            return predictions
        preds_per_doc = {pred.id: pred for pred in predictions}
        for pred in predictions:
            if pred.id not in overlapping_docs:
                continue
            relevant_overlaps = overlapping_docs[pred.id]
            for ans_idx in reversed(range(len(pred.prediction))):
                ans = pred.prediction[ans_idx]
                if ans.answer_type != 'span':
                    continue
                for overlap in relevant_overlaps:
                    if not self._qa_cand_in_overlap(ans, overlap):
                        continue
                    overlapping_doc_pred = preds_per_doc[overlap['doc_id']]
                    cur_doc_overlap = [ol for ol in overlapping_docs[overlap['doc_id']] if ol['doc_id'] == pred.id][0]
                    for pot_dupl_ans_idx in reversed(range(len(overlapping_doc_pred.prediction))):
                        pot_dupl_ans = overlapping_doc_pred.prediction[pot_dupl_ans_idx]
                        if pot_dupl_ans.answer_type != 'span':
                            continue
                        if not self._qa_cand_in_overlap(pot_dupl_ans, cur_doc_overlap):
                            continue
                        if self._is_duplicate_answer(ans, overlap, pot_dupl_ans, cur_doc_overlap):
                            if ans.confidence < pot_dupl_ans.confidence:
                                pred.prediction.pop(ans_idx)
                            else:
                                overlapping_doc_pred.prediction.pop(pot_dupl_ans_idx)
        return predictions

    @staticmethod
    def _is_duplicate_answer(ans: 'QACandidate', ans_overlap: Dict, pot_dupl_ans: 'QACandidate', pot_dupl_ans_overlap: Dict) -> bool:
        if False:
            while True:
                i = 10
        answer_start_in_overlap = ans.offset_answer_start - ans_overlap['range'][0]
        answer_end_in_overlap = ans.offset_answer_end - ans_overlap['range'][0]
        pot_dupl_ans_start_in_overlap = pot_dupl_ans.offset_answer_start - pot_dupl_ans_overlap['range'][0]
        pot_dupl_ans_end_in_overlap = pot_dupl_ans.offset_answer_end - pot_dupl_ans_overlap['range'][0]
        return answer_start_in_overlap == pot_dupl_ans_start_in_overlap and answer_end_in_overlap == pot_dupl_ans_end_in_overlap

    @staticmethod
    def _qa_cand_in_overlap(cand: 'QACandidate', overlap: Dict) -> bool:
        if False:
            return 10
        if cand.offset_answer_start < overlap['range'][0] or cand.offset_answer_end > overlap['range'][1]:
            return False
        return True

    @staticmethod
    def _identify_overlapping_docs(documents: List[Document]) -> Dict[str, List]:
        if False:
            i = 10
            return i + 15
        docs_by_ids = {doc.id: doc for doc in documents}
        overlapping_docs = {}
        for doc in documents:
            if '_split_overlap' not in doc.meta:
                continue
            current_overlaps = [overlap for overlap in doc.meta['_split_overlap'] if overlap['doc_id'] in docs_by_ids]
            if current_overlaps:
                overlapping_docs[doc.id] = current_overlaps
        return overlapping_docs

    def calibrate_confidence_scores(self, document_store: BaseDocumentStore, device: Optional[Union[str, 'torch.device']]=None, label_index: str='label', doc_index: str='eval_document', label_origin: str='gold_label'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calibrates confidence scores on evaluation documents in the DocumentStore.\n\n        :param document_store: DocumentStore containing the evaluation documents\n        :param device: The device on which the tensors should be processed.\n                       Choose from torch.device("cpu") and torch.device("cuda") (or simply "cpu" or "cuda")\n                       or use the Reader\'s device by default.\n        :param label_index: Index/Table name where labeled questions are stored\n        :param doc_index: Index/Table name where documents that are used for evaluation are stored\n        :param label_origin: Field name where the gold labels are stored\n        '
        if device is None:
            device = self.devices[0]
        self.eval(document_store=document_store, device=device, label_index=label_index, doc_index=doc_index, label_origin=label_origin, calibrate_conf_scores=True)

    @staticmethod
    def _check_no_answer(c: 'QACandidate'):
        if False:
            while True:
                i = 10
        if c.offset_answer_start == 0 and c.offset_answer_end == 0 and (c.answer != 'no_answer'):
            logger.error("Invalid 'no_answer': Got a prediction for position 0, but answer string is not 'no_answer'")
        return c.answer == 'no_answer'

    def predict_on_texts(self, question: str, texts: List[str], top_k: Optional[int]=None):
        if False:
            return 10
        '\n        Use loaded QA model to find answers for a question in the supplied list of Document.\n        Returns dictionaries containing answers sorted by (desc.) score.\n        Example:\n\n         ```python\n         {\n             \'question\': \'Who is the father of Arya Stark?\',\n             \'answers\':[\n                          {\'answer\': \'Eddard,\',\n                          \'context\': " She travels with her father, Eddard, to King\'s Landing when he is ",\n                          \'offset_answer_start\': 147,\n                          \'offset_answer_end\': 154,\n                          \'score\': 0.9787139466668613,\n                          \'document_id\': \'1337\'\n                          },...\n                       ]\n         }\n         ```\n\n        :param question: Question string\n        :param texts: A list of Document texts as a string type\n        :param top_k: The maximum number of answers to return\n        :return: Dict containing question and answers\n        '
        documents = []
        for text in texts:
            documents.append(Document(content=text))
        predictions = self.predict(question, documents, top_k)
        return predictions

    @classmethod
    def convert_to_onnx(cls, model_name: str, output_path: Path, convert_to_float16: bool=False, quantize: bool=False, task_type: str='question_answering', opset_version: int=11):
        if False:
            return 10
        '\n        Convert a PyTorch BERT model to ONNX format and write to ./onnx-export dir. The converted ONNX model\n        can be loaded with in the `FARMReader` using the export path as `model_name_or_path` param.\n\n        Usage:\n\n            `from haystack.reader.farm import FARMReader\n            from pathlib import Path\n            onnx_model_path = Path("roberta-onnx-model")\n            FARMReader.convert_to_onnx(model_name="deepset/bert-base-cased-squad2", output_path=onnx_model_path)\n            reader = FARMReader(onnx_model_path)`\n\n        :param model_name: transformers model name\n        :param output_path: Path to output the converted model\n        :param convert_to_float16: Many models use float32 precision by default. With the half precision of float16,\n                                   inference is faster on Nvidia GPUs with Tensor core like T4 or V100. On older GPUs,\n                                   float32 could still be be more performant.\n        :param quantize: convert floating point number to integers\n        :param task_type: Type of task for the model. Available options: "question_answering" or "embeddings".\n        :param opset_version: ONNX opset version\n        '
        AdaptiveModel.convert_to_onnx(model_name=model_name, output_path=output_path, task_type=task_type, convert_to_float16=convert_to_float16, quantize=quantize, opset_version=opset_version)