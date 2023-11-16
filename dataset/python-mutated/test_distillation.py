import torch
from haystack.nodes import FARMReader
from haystack.modeling.data_handler.processor import UnlabeledTextProcessor

def create_checkpoint(model):
    if False:
        return 10
    weights = []
    for (name, weight) in model.inferencer.model.named_parameters():
        if 'weight' in name and weight.requires_grad:
            weights.append(torch.clone(weight))
    return weights

def assert_weight_change(weights, new_weights):
    if False:
        print('Hello World!')
    print([torch.equal(old_weight, new_weight) for (old_weight, new_weight) in zip(weights, new_weights)])
    assert not any((torch.equal(old_weight, new_weight) for (old_weight, new_weight) in zip(weights, new_weights)))

def test_prediction_layer_distillation(samples_path):
    if False:
        return 10
    student = FARMReader(model_name_or_path='prajjwal1/bert-mini', num_processes=0)
    teacher = FARMReader(model_name_or_path='prajjwal1/bert-small', num_processes=0)
    student_weights = create_checkpoint(student)
    assert len(student_weights) == 38
    student_weights.pop(-2)
    student.distil_prediction_layer_from(teacher, data_dir=samples_path / 'squad', train_filename='tiny.json')
    new_student_weights = create_checkpoint(student)
    assert len(new_student_weights) == 38
    new_student_weights.pop(-2)
    assert_weight_change(student_weights, new_student_weights)

def test_intermediate_layer_distillation(samples_path):
    if False:
        while True:
            i = 10
    student = FARMReader(model_name_or_path='huawei-noah/TinyBERT_General_4L_312D')
    teacher = FARMReader(model_name_or_path='bert-base-uncased')
    student_weights = create_checkpoint(student)
    assert len(student_weights) == 38
    student_weights.pop(-1)
    student_weights.pop(-1)
    student.distil_intermediate_layers_from(teacher_model=teacher, data_dir=samples_path / 'squad', train_filename='tiny.json')
    new_student_weights = create_checkpoint(student)
    assert len(new_student_weights) == 38
    new_student_weights.pop(-1)
    new_student_weights.pop(-1)
    assert_weight_change(student_weights, new_student_weights)

def test_intermediate_layer_distillation_from_scratch(samples_path):
    if False:
        print('Hello World!')
    student = FARMReader(model_name_or_path='huawei-noah/TinyBERT_General_4L_312D')
    teacher = FARMReader(model_name_or_path='bert-base-uncased')
    student_weights = create_checkpoint(student)
    assert len(student_weights) == 38
    student_weights.pop(-1)
    student_weights.pop(-1)
    processor = UnlabeledTextProcessor(tokenizer=teacher.inferencer.processor.tokenizer, max_seq_len=128, train_filename='doc_2.txt', data_dir=samples_path / 'docs')
    student.distil_intermediate_layers_from(teacher_model=teacher, data_dir=samples_path / 'squad', train_filename='tiny.json', processor=processor)
    new_student_weights = create_checkpoint(student)
    assert len(new_student_weights) == 38
    new_student_weights.pop(-1)
    new_student_weights.pop(-1)
    assert_weight_change(student_weights, new_student_weights)