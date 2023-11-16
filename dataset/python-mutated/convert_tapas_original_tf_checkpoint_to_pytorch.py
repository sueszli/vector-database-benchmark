"""Convert TAPAS checkpoint."""
import argparse
from transformers import TapasConfig, TapasForMaskedLM, TapasForQuestionAnswering, TapasForSequenceClassification, TapasModel, TapasTokenizer, load_tf_weights_in_tapas
from transformers.utils import logging
logging.set_verbosity_info()

def convert_tf_checkpoint_to_pytorch(task, reset_position_index_per_cell, tf_checkpoint_path, tapas_config_file, pytorch_dump_path):
    if False:
        while True:
            i = 10
    config = TapasConfig.from_json_file(tapas_config_file)
    config.reset_position_index_per_cell = reset_position_index_per_cell
    if task == 'SQA':
        model = TapasForQuestionAnswering(config=config)
    elif task == 'WTQ':
        config.num_aggregation_labels = 4
        config.use_answer_as_supervision = True
        config.answer_loss_cutoff = 0.664694
        config.cell_selection_preference = 0.207951
        config.huber_loss_delta = 0.121194
        config.init_cell_selection_weights_to_zero = True
        config.select_one_column = True
        config.allow_empty_column_selection = False
        config.temperature = 0.0352513
        model = TapasForQuestionAnswering(config=config)
    elif task == 'WIKISQL_SUPERVISED':
        config.num_aggregation_labels = 4
        config.use_answer_as_supervision = False
        config.answer_loss_cutoff = 36.4519
        config.cell_selection_preference = 0.903421
        config.huber_loss_delta = 222.088
        config.init_cell_selection_weights_to_zero = True
        config.select_one_column = True
        config.allow_empty_column_selection = True
        config.temperature = 0.763141
        model = TapasForQuestionAnswering(config=config)
    elif task == 'TABFACT':
        model = TapasForSequenceClassification(config=config)
    elif task == 'MLM':
        model = TapasForMaskedLM(config=config)
    elif task == 'INTERMEDIATE_PRETRAINING':
        model = TapasModel(config=config)
    else:
        raise ValueError(f'Task {task} not supported.')
    print(f'Building PyTorch model from configuration: {config}')
    load_tf_weights_in_tapas(model, config, tf_checkpoint_path)
    print(f'Save PyTorch model to {pytorch_dump_path}')
    model.save_pretrained(pytorch_dump_path)
    print(f'Save tokenizer files to {pytorch_dump_path}')
    tokenizer = TapasTokenizer(vocab_file=tf_checkpoint_path[:-10] + 'vocab.txt', model_max_length=512)
    tokenizer.save_pretrained(pytorch_dump_path)
    print('Used relative position embeddings:', model.config.reset_position_index_per_cell)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='SQA', type=str, help='Model task for which to convert a checkpoint. Defaults to SQA.')
    parser.add_argument('--reset_position_index_per_cell', default=False, action='store_true', help='Whether to use relative position embeddings or not. Defaults to True.')
    parser.add_argument('--tf_checkpoint_path', default=None, type=str, required=True, help='Path to the TensorFlow checkpoint path.')
    parser.add_argument('--tapas_config_file', default=None, type=str, required=True, help='The config json file corresponding to the pre-trained TAPAS model. \nThis specifies the model architecture.')
    parser.add_argument('--pytorch_dump_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.task, args.reset_position_index_per_cell, args.tf_checkpoint_path, args.tapas_config_file, args.pytorch_dump_path)