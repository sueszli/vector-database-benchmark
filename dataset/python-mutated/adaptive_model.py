import copy
import json
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Iterable, Dict, Union, List, Optional, Callable
import numpy
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForQuestionAnswering
from transformers.convert_graph_to_onnx import convert, quantize as quantize_model
from haystack.modeling.data_handler.processor import Processor
from haystack.modeling.model.language_model import get_language_model, LanguageModel, _get_model_type, capitalize_model_type
from haystack.modeling.model.prediction_head import PredictionHead, QuestionAnsweringHead
from haystack.utils.experiment_tracking import Tracker as tracker
logger = logging.getLogger(__name__)

class BaseAdaptiveModel:
    """
    Base Class for implementing AdaptiveModel with frameworks like PyTorch and ONNX.
    """
    language_model: LanguageModel
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        This automatically keeps track of all available subclasses.\n        Enables generic load() for all specific AdaptiveModel implementation.\n        '
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self, prediction_heads: Union[List[PredictionHead], nn.ModuleList]):
        if False:
            i = 10
            return i + 15
        self.prediction_heads = prediction_heads

    @classmethod
    def load(cls, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Load corresponding AdaptiveModel Class(AdaptiveModel/ONNXAdaptiveModel) based on the\n        files in the load_dir.\n\n        :param kwargs: Arguments to pass for loading the model.\n        :return: Instance of a model.\n        '
        if (Path(kwargs['load_dir']) / 'model.onnx').is_file():
            model = cls.subclasses['ONNXAdaptiveModel'].load(**kwargs)
        else:
            model = cls.subclasses['AdaptiveModel'].load(**kwargs)
        return model

    def logits_to_preds(self, logits: torch.Tensor, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Get predictions from all prediction heads.\n\n        :param logits: Logits that can vary in shape and type, depending on task.\n        :return: A list of all predictions from all prediction heads.\n        '
        all_preds = []
        for (head, logits_for_head) in zip(self.prediction_heads, logits):
            preds = head.logits_to_preds(logits=logits_for_head, **kwargs)
            all_preds.append(preds)
        return all_preds

    def formatted_preds(self, logits: torch.Tensor, **kwargs):
        if False:
            return 10
        '\n        Format predictions for inference.\n\n        :param logits: Model logits.\n        :return: Predictions in the right format.\n        '
        n_heads = len(self.prediction_heads)
        if n_heads == 0:
            preds_final = self.language_model.formatted_preds(logits=logits, **kwargs)
        elif n_heads == 1:
            preds_final = []
            try:
                preds = kwargs['preds']
                temp = [y[0] for y in preds]
                preds_flat = [item for sublist in temp for item in sublist]
                kwargs['preds'] = preds_flat
            except KeyError:
                kwargs['preds'] = None
            head = self.prediction_heads[0]
            logits_for_head = logits[0]
            preds = head.formatted_preds(logits=logits_for_head, **kwargs)
            if type(preds) == list:
                preds_final += preds
            elif type(preds) == dict and 'predictions' in preds:
                preds_final.append(preds)
        return preds_final

    def connect_heads_with_processor(self, tasks: Dict, require_labels: bool=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Populates prediction head with information coming from tasks.\n\n        :param tasks: A dictionary where the keys are the names of the tasks and\n                      the values are the details of the task (e.g. label_list, metric,\n                      tensor name).\n        :param require_labels: If True, an error will be thrown when a task is\n                               not supplied with labels.\n        :return: None\n        '
        for head in self.prediction_heads:
            head.label_tensor_name = tasks[head.task_name]['label_tensor_name']
            label_list = tasks[head.task_name]['label_list']
            if not label_list and require_labels:
                raise Exception(f"The task '{head.task_name}' is missing a valid set of labels")
            label_list = tasks[head.task_name]['label_list']
            head.label_list = label_list
            head.metric = tasks[head.task_name]['metric']

    @classmethod
    def _get_prediction_head_files(cls, load_dir: Union[str, Path], strict: bool=True):
        if False:
            i = 10
            return i + 15
        load_dir = Path(load_dir)
        files = os.listdir(load_dir)
        model_files = [load_dir / f for f in files if '.bin' in f and 'prediction_head' in f]
        config_files = [load_dir / f for f in files if 'config.json' in f and 'prediction_head' in f]
        model_files.sort()
        config_files.sort()
        if strict:
            error_str = f'There is a mismatch in number of model files ({len(model_files)}) and config files ({len(config_files)}).This might be because the Language Model Prediction Head does not currently support saving and loading'
            assert len(model_files) == len(config_files), error_str
        logger.info('Found files for loading %s prediction heads', len(model_files))
        return (model_files, config_files)

def loss_per_head_sum(loss_per_head: Iterable, global_step: Optional[int]=None, batch: Optional[Dict]=None):
    if False:
        print('Hello World!')
    '\n    Sums up the loss of each prediction head.\n\n    :param loss_per_head: List of losses.\n    '
    return sum(loss_per_head)

class AdaptiveModel(nn.Module, BaseAdaptiveModel):
    """
    PyTorch implementation containing all the modelling needed for your NLP task. Combines a language
    model and a prediction head. Allows for gradient flow back to the language model component.
    """

    def __init__(self, language_model: LanguageModel, prediction_heads: List[PredictionHead], embeds_dropout_prob: float, lm_output_types: Union[str, List[str]], device: torch.device, loss_aggregation_fn: Optional[Callable]=None):
        if False:
            return 10
        '\n        :param language_model: Any model that turns token ids into vector representations.\n        :param prediction_heads: A list of models that take embeddings and return logits for a given task.\n        :param embeds_dropout_prob: The probability that a value in the embeddings returned by the\n                                    language model will be zeroed.\n        :param lm_output_types: How to extract the embeddings from the final layer of the language model. When set\n                                to "per_token", one embedding will be extracted per input token. If set to\n                                "per_sequence", a single embedding will be extracted to represent the full\n                                input sequence. Can either be a single string, or a list of strings,\n                                one for each prediction head.\n        :param device: The device on which this model will operate. Either torch.device("cpu") or torch.device("cuda").\n        :param loss_aggregation_fn: Function to aggregate the loss of multiple prediction heads.\n                                    Input: loss_per_head (list of tensors), global_step (int), batch (dict)\n                                    Output: aggregated loss (tensor)\n                                    Default is a simple sum:\n                                    `lambda loss_per_head, global_step=None, batch=None: sum(tensors)`\n                                    However, you can pass more complex functions that depend on the\n                                    current step (e.g. for round-robin style multitask learning) or the actual\n                                    content of the batch (e.g. certain labels)\n                                    Note: The loss at this stage is per sample, i.e one tensor of\n                                    shape (batchsize) per prediction head.\n        '
        super(AdaptiveModel, self).__init__()
        self.device = device
        self.language_model = language_model.to(device)
        self.lm_output_dims = language_model.output_dims
        self.prediction_heads = nn.ModuleList([ph.to(device) for ph in prediction_heads])
        self.fit_heads_to_lm()
        self.dropout = nn.Dropout(embeds_dropout_prob)
        self.lm_output_types = [lm_output_types] if isinstance(lm_output_types, str) else lm_output_types
        self.log_params()
        if not loss_aggregation_fn:
            loss_aggregation_fn = loss_per_head_sum
        self.loss_aggregation_fn = loss_aggregation_fn

    def fit_heads_to_lm(self):
        if False:
            while True:
                i = 10
        "\n        This iterates over each prediction head and ensures that its input\n        dimensionality matches the output dimensionality of the language model.\n        If it doesn't, it is resized so it does fit.\n        "
        for ph in self.prediction_heads:
            ph.resize_input(self.lm_output_dims)
            ph.to(self.device)

    def bypass_ph(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Replaces methods in the prediction heads with dummy functions.\n        Used for benchmarking where we want to isolate the LanguageModel run time\n        from the PredictionHead run time.\n        '

        def fake_forward(x):
            if False:
                print('Hello World!')
            '\n            Slices lm vector outputs of shape (batch_size, max_seq_len, dims) --> (batch_size, max_seq_len, 2)\n            '
            return x.narrow(2, 0, 2)

        def fake_logits_to_preds(logits, **kwargs):
            if False:
                print('Hello World!')
            batch_size = logits.shape[0]
            return [None, None] * batch_size

        def fake_formatted_preds(**kwargs):
            if False:
                print('Hello World!')
            return None
        for ph in self.prediction_heads:
            ph.forward = fake_forward
            ph.logits_to_preds = fake_logits_to_preds
            ph.formatted_preds = fake_formatted_preds

    def save(self, save_dir: Union[str, Path]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Saves the language model and prediction heads. This will generate a config file\n        and model weights for each.\n\n        :param save_dir: Path to save the AdaptiveModel to.\n        '
        os.makedirs(save_dir, exist_ok=True)
        self.language_model.save(save_dir)
        for (i, ph) in enumerate(self.prediction_heads):
            ph.save(save_dir, i)

    @classmethod
    def load(cls, load_dir: Union[str, Path], device: Union[str, torch.device], strict: bool=True, processor: Optional[Processor]=None):
        if False:
            while True:
                i = 10
        '\n        Loads an AdaptiveModel from a directory. The directory must contain:\n\n        * language_model.bin\n        * language_model_config.json\n        * prediction_head_X.bin  multiple PH possible\n        * prediction_head_X_config.json\n        * processor_config.json config for transforming input\n        * vocab.txt vocab file for language model, turning text to Wordpiece Tokens\n\n        :param load_dir: Location where the AdaptiveModel is stored.\n        :param device: Specifies the device to which you want to send the model, either torch.device("cpu") or torch.device("cuda").\n        :param strict: Whether to strictly enforce that the keys loaded from saved model match the ones in\n                       the PredictionHead (see torch.nn.module.load_state_dict()).\n        :param processor: Processor to populate prediction head with information coming from tasks.\n        '
        device = torch.device(device)
        language_model = get_language_model(load_dir)
        (_, ph_config_files) = cls._get_prediction_head_files(load_dir)
        prediction_heads = []
        ph_output_type = []
        for config_file in ph_config_files:
            head = PredictionHead.load(config_file, strict=strict)
            prediction_heads.append(head)
            ph_output_type.append(head.ph_output_type)
        model = cls(language_model, prediction_heads, 0.1, ph_output_type, device)
        if processor:
            model.connect_heads_with_processor(processor.tasks)
        return model

    @classmethod
    def convert_from_transformers(cls, model_name_or_path, device: Union[str, torch.device], revision: Optional[str]=None, task_type: str='question_answering', processor: Optional[Processor]=None, use_auth_token: Optional[Union[bool, str]]=None, **kwargs) -> 'AdaptiveModel':
        if False:
            while True:
                i = 10
        '\n        Load a (downstream) model from huggingface\'s transformers format. Use cases:\n         - continue training in Haystack (e.g. take a squad QA model and fine-tune on your own data)\n         - compare models without switching frameworks\n         - use model directly for inference\n\n        :param model_name_or_path: local path of a saved model or name of a public one.\n                                              Exemplary public names:\n                                              - distilbert-base-uncased-distilled-squad\n                                              - deepset/bert-large-uncased-whole-word-masking-squad2\n\n                                              See https://huggingface.co/models for full list\n        :param device: torch.device("cpu") or torch.device("cuda")\n        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.\n                         Right now accepts only \'question_answering\'.\n        :param processor: populates prediction head with information coming from tasks.\n        :param use_auth_token: The API token used to download private models from Huggingface.\n                               If this parameter is set to `True`, then the token generated when running\n                               `transformers-cli login` (stored in ~/.huggingface) will be used.\n                               Additional information can be found here\n                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained\n        :return: AdaptiveModel\n        '
        lm = get_language_model(model_name_or_path, revision=revision, use_auth_token=use_auth_token, model_kwargs=kwargs)
        if task_type is None:
            architecture = lm.model.config.architectures[0]
            if 'QuestionAnswering' in architecture:
                task_type = 'question_answering'
            else:
                logger.error("Could not infer task type from model config. Please provide task type manually. ('question_answering' or 'embeddings')")
        if task_type == 'question_answering':
            ph = QuestionAnsweringHead.load(model_name_or_path, revision=revision, use_auth_token=use_auth_token, **kwargs)
            adaptive_model = cls(language_model=lm, prediction_heads=[ph], embeds_dropout_prob=0.1, lm_output_types='per_token', device=device)
        elif task_type == 'embeddings':
            adaptive_model = cls(language_model=lm, prediction_heads=[], embeds_dropout_prob=0.1, lm_output_types=['per_token', 'per_sequence'], device=device)
        if processor:
            adaptive_model.connect_heads_with_processor(processor.tasks)
        return adaptive_model

    def convert_to_transformers(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Convert an adaptive model to huggingface's transformers format. Returns a list containing one model for each\n        prediction head.\n\n        :return: List of huggingface transformers models.\n        "
        converted_models = []
        for prediction_head in self.prediction_heads:
            if len(prediction_head.layer_dims) != 2:
                logger.error('Currently conversion only works for PredictionHeads that are a single layer Feed Forward NN with dimensions [LM_output_dim, number_classes].\n            Your PredictionHead has %s dimensions.', str(prediction_head.layer_dims))
                continue
            if prediction_head.model_type == 'span_classification':
                transformers_model = self._convert_to_transformers_qa(prediction_head)
                converted_models.append(transformers_model)
            else:
                logger.error('Haystack -> Transformers conversion is not supported yet for prediction heads of type %s', prediction_head.model_type)
        return converted_models

    def _convert_to_transformers_qa(self, prediction_head):
        if False:
            i = 10
            return i + 15
        self.language_model.model.pooler = None
        transformers_model = AutoModelForQuestionAnswering.from_config(self.language_model.model.config)
        setattr(transformers_model, transformers_model.base_model_prefix, self.language_model.model)
        transformers_model.qa_outputs.load_state_dict(prediction_head.feed_forward.feed_forward[0].state_dict())
        return transformers_model

    def logits_to_loss_per_head(self, logits: torch.Tensor, **kwargs):
        if False:
            print('Hello World!')
        '\n        Collect losses from each prediction head.\n\n        :param logits: Logits, can vary in shape and type, depending on task.\n        :return: The per sample per prediciton head loss whose first two dimensions\n                 have length n_pred_heads, batch_size.\n        '
        all_losses = []
        for (head, logits_for_one_head) in zip(self.prediction_heads, logits):
            assert hasattr(head, 'label_tensor_name'), f"Label_tensor_names are missing inside the {head.task_name} Prediction Head. Did you connect the model with the processor through either 'model.connect_heads_with_processor(processor.tasks)' or by passing the processor to the Adaptive Model?"
            all_losses.append(head.logits_to_loss(logits=logits_for_one_head, **kwargs))
        return all_losses

    def logits_to_loss(self, logits: torch.Tensor, global_step: Optional[int]=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Get losses from all prediction heads & reduce to single loss *per sample*.\n\n        :param logits: Logits, can vary in shape and type, depending on task.\n        :param global_step: Number of current training step.\n        :param kwargs: Placeholder for passing generic parameters.\n                       Note: Contains the batch (as dict of tensors), when called from Trainer.train().\n        :return: torch.tensor that is the per sample loss (len: batch_size)\n        '
        all_losses = self.logits_to_loss_per_head(logits, **kwargs)
        loss = self.loss_aggregation_fn(all_losses, global_step=global_step, batch=kwargs)
        return loss

    def prepare_labels(self, **kwargs):
        if False:
            return 10
        '\n        Label conversion to original label space, per prediction head.\n\n        :param label_maps: dictionary for mapping ids to label strings\n        :type label_maps: dict[int:str]\n        :return: labels in the right format\n        '
        all_labels = []
        for head in self.prediction_heads:
            labels = head.prepare_labels(**kwargs)
            all_labels.append(labels)
        return all_labels

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor, padding_mask: torch.Tensor, output_hidden_states: bool=False, output_attentions: bool=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Push data through the whole model and returns logits. The data will\n        propagate through the language model and each of the attached prediction heads.\n\n        :param input_ids: The IDs of each token in the input sequence. It's a tensor of shape [batch_size, max_seq_len].\n        :param segment_ids: The ID of the segment. For example, in next sentence prediction, the tokens in the\n           first sentence are marked with 0 and the tokens in the second sentence are marked with 1.\n           It is a tensor of shape [batch_size, max_seq_len].\n        :param padding_mask: A mask that assigns 1 to valid input tokens and 0 to padding tokens\n           of shape [batch_size, max_seq_len].\n        :param output_hidden_states: Whether to output hidden states\n        :param output_attentions: Whether to output attentions\n        :return: All logits as torch.tensor or multiple tensors.\n        "
        output_tuple = self.language_model.forward(input_ids=input_ids, segment_ids=segment_ids, attention_mask=padding_mask, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        if output_hidden_states and output_attentions:
            (sequence_output, pooled_output, hidden_states, attentions) = output_tuple
        elif output_hidden_states:
            (sequence_output, pooled_output, hidden_states) = output_tuple
        elif output_attentions:
            (sequence_output, pooled_output, attentions) = output_tuple
        else:
            (sequence_output, pooled_output) = output_tuple
        all_logits = []
        if len(self.prediction_heads) > 0:
            for (head, lm_out) in zip(self.prediction_heads, self.lm_output_types):
                if lm_out == 'per_token':
                    output = self.dropout(sequence_output)
                elif lm_out == 'per_sequence' or lm_out == 'per_sequence_continuous':
                    output = self.dropout(pooled_output)
                elif lm_out == 'per_token_squad':
                    output = self.dropout(sequence_output)
                else:
                    raise ValueError('Unknown extraction strategy from language model: {}'.format(lm_out))
                all_logits.append(head(output))
        else:
            all_logits.append((sequence_output, pooled_output))
        if output_hidden_states and output_attentions:
            return (all_logits, hidden_states, attentions)
        if output_hidden_states:
            return (all_logits, hidden_states)
        if output_attentions:
            return (all_logits, attentions)
        return all_logits

    def forward_lm(self, **kwargs):
        if False:
            return 10
        '\n        Forward pass for the language model.\n\n        :return: Tuple containing list of embeddings for each token and\n                 embedding for whole sequence.\n        '
        try:
            extraction_layer = self.language_model.extraction_layer
        except:
            extraction_layer = -1
        if extraction_layer == -1:
            (sequence_output, pooled_output) = self.language_model(**kwargs, return_dict=False, output_all_encoded_layers=False)
        else:
            self.language_model.enable_hidden_states_output()
            (sequence_output, pooled_output, all_hidden_states) = self.language_model(**kwargs, return_dict=False)
            sequence_output = all_hidden_states[extraction_layer]
            pooled_output = None
            self.language_model.disable_hidden_states_output()
        return (sequence_output, pooled_output)

    def log_params(self):
        if False:
            print('Hello World!')
        '\n        Logs parameters to generic logger MlLogger\n        '
        params = {'lm_type': self.language_model.__class__.__name__, 'lm_name': self.language_model.name, 'prediction_heads': ','.join([head.__class__.__name__ for head in self.prediction_heads]), 'lm_output_types': ','.join(self.lm_output_types)}
        try:
            tracker.track_params(params)
        except Exception as e:
            logger.warning("ML logging didn't work: %s", e)

    def verify_vocab_size(self, vocab_size: int):
        if False:
            return 10
        '\n        Verifies that the model fits to the tokenizer vocabulary.\n        They could diverge in case of custom vocabulary added via tokenizer.add_tokens()\n        '
        model_vocab_len = self.language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
        msg = f"Vocab size of tokenizer {vocab_size} doesn't match with model {model_vocab_len}. If you added a custom vocabulary to the tokenizer, make sure to supply 'n_added_tokens' to get_language_model() and BertStyleLM.load()"
        assert vocab_size == model_vocab_len, msg
        for head in self.prediction_heads:
            if head.model_type == 'language_modelling':
                ph_decoder_len = head.decoder.weight.shape[0]
                assert vocab_size == ph_decoder_len, msg

    def get_language(self):
        if False:
            while True:
                i = 10
        return self.language_model.language

    @classmethod
    def convert_to_onnx(cls, model_name: str, output_path: Path, task_type: str, convert_to_float16: bool=False, quantize: bool=False, opset_version: int=11, use_auth_token: Optional[Union[str, bool]]=None):
        if False:
            i = 10
            return i + 15
        '\n        Convert a PyTorch model from transformers hub to an ONNX Model.\n\n        :param model_name: Transformers model name.\n        :param output_path: Output Path to write the converted model to.\n        :param task_type: Type of task for the model. Available options: "question_answering"\n        :param convert_to_float16: By default, the model uses float32 precision. With half precision of float16, inference\n                                   should be faster on Nvidia GPUs with Tensor core like T4 or V100. On older GPUs, float32\n                                   might be more performant.\n        :param quantize: Convert floating point number to integers\n        :param opset_version: ONNX opset version.\n        :param use_auth_token: The API token used to download private models from Huggingface.\n                               If this parameter is set to `True`, then the token generated when running\n                               `transformers-cli login` (stored in ~/.huggingface) will be used.\n                               Additional information can be found here\n                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained\n        :return: None.\n        '
        model_type = capitalize_model_type(_get_model_type(model_name))
        if model_type not in ['Bert', 'Roberta', 'XLMRoberta']:
            raise Exception("The current ONNX conversion only support 'BERT', 'RoBERTa', and 'XLMRoberta' models.")
        task_type_to_pipeline_map = {'question_answering': 'question-answering'}
        convert(pipeline_name=task_type_to_pipeline_map[task_type], framework='pt', model=model_name, output=output_path / 'model.onnx', opset=opset_version, use_external_format=model_type == 'XLMRoberta', use_auth_token=use_auth_token)
        processor = Processor.convert_from_transformers(tokenizer_name_or_path=model_name, task_type=task_type, max_seq_len=256, doc_stride=128, use_fast=True, use_auth_token=use_auth_token)
        processor.save(output_path)
        model = AdaptiveModel.convert_from_transformers(model_name, device=torch.device('cpu'), task_type=task_type, use_auth_token=use_auth_token)
        model.save(output_path)
        os.remove(output_path / 'language_model.bin')
        onnx_model_config = {'task_type': task_type, 'onnx_opset_version': opset_version, 'language_model_class': model_type, 'language': model.language_model.language}
        with open(output_path / 'onnx_model_config.json', 'w') as f:
            json.dump(onnx_model_config, f)
        if convert_to_float16:
            from onnxruntime_tools import optimizer
            config = AutoConfig.from_pretrained(model_name, use_auth_token=use_auth_token)
            optimized_model = optimizer.optimize_model(input=str(output_path / 'model.onnx'), model_type='bert', num_heads=config.num_hidden_layers, hidden_size=config.hidden_size)
            optimized_model.convert_model_float32_to_float16()
            optimized_model.save_model_to_file('model.onnx')
        if quantize:
            quantize_model(output_path / 'model.onnx')

class ONNXAdaptiveModel(BaseAdaptiveModel):
    """
    Implementation of ONNX Runtime for Inference of ONNX Models.

    Existing PyTorch based Haystack.basics AdaptiveModel can be converted to ONNX format using AdaptiveModel.convert_to_onnx().
    The conversion is currently only implemented for Question Answering Models.

    For inference, this class is compatible with the Haystack.basics Inferencer.
    """

    def __init__(self, onnx_session, language_model_class: str, language: str, prediction_heads: List[PredictionHead], device: torch.device):
        if False:
            while True:
                i = 10
        '\n        :param onnx_session: ? # TODO\n        :param language_model_class: Class of LanguageModel\n        :param language: Language the model is trained for.\n        :param prediction_heads: A list of models that take embeddings and return logits for a given task.\n        :param device: The device on which this model will operate. Either torch.device("cpu") or torch.device("cuda").\n        '
        import onnxruntime
        super().__init__(prediction_heads)
        if str(device) == 'cuda' and onnxruntime.get_device() != 'GPU':
            raise Exception(f'Device {device} not available for Inference. For CPU, run pip install onnxruntime andfor GPU run pip install onnxruntime-gpu')
        self.onnx_session = onnx_session
        self.language_model_class = language_model_class
        self.language = language
        self.prediction_heads = prediction_heads
        self.device = device

    @classmethod
    def load(cls, load_dir: Union[str, Path], device: Union[str, torch.device], **kwargs):
        if False:
            print('Hello World!')
        '\n        Loads an ONNXAdaptiveModel from a directory.\n\n        :param load_dir: Location where the ONNXAdaptiveModel is stored.\n        :param device: The device on which this model will operate. Either torch.device("cpu") or torch.device("cuda").\n        '
        device = torch.device(device)
        load_dir = Path(load_dir)
        import onnxruntime
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.intra_op_num_threads = multiprocessing.cpu_count()
        providers = kwargs.get('providers', ['CPUExecutionProvider'] if device.type == 'cpu' else ['CUDAExecutionProvider'])
        onnx_session = onnxruntime.InferenceSession(str(load_dir / 'model.onnx'), sess_options, providers=providers)
        (_, ph_config_files) = cls._get_prediction_head_files(load_dir, strict=False)
        prediction_heads = []
        ph_output_type = []
        for config_file in ph_config_files:
            head = PredictionHead.load(config_file, load_weights=False)
            prediction_heads.append(head)
            ph_output_type.append(head.ph_output_type)
        with open(load_dir / 'onnx_model_config.json') as f:
            model_config = json.load(f)
            language_model_class = model_config['language_model_class']
            language = model_config['language']
        return cls(onnx_session, language_model_class, language, prediction_heads, device)

    def forward(self, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Perform forward pass on the model and return the logits.\n\n        :param kwargs: All arguments that need to be passed on to the model.\n        :return: All logits as torch.tensor or multiple tensors.\n        '
        with torch.inference_mode():
            if self.language_model_class == 'Bert':
                input_to_onnx = {'input_ids': numpy.ascontiguousarray(kwargs['input_ids'].cpu().numpy()), 'attention_mask': numpy.ascontiguousarray(kwargs['padding_mask'].cpu().numpy()), 'token_type_ids': numpy.ascontiguousarray(kwargs['segment_ids'].cpu().numpy())}
            elif self.language_model_class in ['Roberta', 'XLMRoberta']:
                input_to_onnx = {'input_ids': numpy.ascontiguousarray(kwargs['input_ids'].cpu().numpy()), 'attention_mask': numpy.ascontiguousarray(kwargs['padding_mask'].cpu().numpy())}
            res = self.onnx_session.run(None, input_to_onnx)
            res = numpy.stack(res).transpose(1, 2, 0)
            logits = [torch.Tensor(res).to(self.device)]
        return logits

    def eval(self):
        if False:
            print('Hello World!')
        '\n        Stub to make ONNXAdaptiveModel compatible with the PyTorch AdaptiveModel.\n        '
        return True

    def get_language(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the language(s) the model was trained for.\n        :return: str\n        '
        return self.language

class ONNXWrapper(AdaptiveModel):
    """
    Wrapper Class for converting PyTorch models to ONNX.

    As of torch v1.4.0, torch.onnx.export only support passing positional arguments
    to the forward pass of the model. However, the AdaptiveModel's forward takes keyword arguments.
    This class circumvents the issue by converting positional arguments to keyword arguments.
    """

    @classmethod
    def load_from_adaptive_model(cls, adaptive_model: AdaptiveModel):
        if False:
            while True:
                i = 10
        model = copy.deepcopy(adaptive_model)
        model.__class__ = ONNXWrapper
        return model

    def forward(self, *batch):
        if False:
            return 10
        return super().forward(input_ids=batch[0], padding_mask=batch[1], segment_ids=batch[2])