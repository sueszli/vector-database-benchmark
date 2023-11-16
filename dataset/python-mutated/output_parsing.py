import logging
import json5 as json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Optional
from .prompt_template import PromptTemplate
from .encoding import Encoding
logger = logging.getLogger(__name__)
DEFAULT_INDENT = 2

def flatten_outputs(input_path: str, output_path: str, stability_value: int=1):
    if False:
        return 10
    'Flatten batched outputs from JobManager into a format where each line is a single example.'
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        output_list = []
        for (line_idx, line) in enumerate(f_in):
            if len(line.strip()) == 0:
                continue
            job = dict(json.loads(line.strip()))
            logger.info(f"Processing job found on line #{line_idx} containing inputs: {job['input_idx']}.")
            if 'output_examples' not in job:
                logger.info(f"Couldn't find output_examples in job found on line #{line_idx}.")
                continue
            output_examples: List[List[dict]] = [sample for sample in job['output_examples'] if sample is not None]
            output_examples = [list(sample) for sample in zip(*output_examples)]
            for (input_idx, input_example, output_example) in zip(job['input_idx'], job['input_examples'], output_examples):
                example_obj = job.copy()
                example_obj['input_idx'] = input_idx
                example_obj['input_examples'] = input_example
                example_obj['output_examples'] = output_example
                example_obj['input_example'] = example_obj.pop('input_examples')
                example_obj['parsed_output_samples'] = example_obj.pop('output_examples')
                output_list.append(example_obj)
        for output in output_list:
            stabilized_parsed_output_samples = []
            for sample_batch in batch_list(output['parsed_output_samples'], stability_value):
                label_list = defaultdict(list)
                sample_batch_outputs = {}
                for parsed_results in sample_batch:
                    for label in parsed_results:
                        label_list[label].append(parsed_results[label])
                for (label, values) in label_list.items():
                    majority_value = get_majority_value(values)
                    sample_batch_outputs[label] = majority_value
                stabilized_parsed_output_samples.append(sample_batch_outputs)
            output['parsed_output_samples'] = stabilized_parsed_output_samples
        output_list = sorted(output_list, key=lambda x: x['input_idx'])
        for example_obj in output_list:
            f_out.write(json.dumps(example_obj, quote_keys=True) + '\n')

def decode_example(example: str, label_keys: List[str], encoding: Encoding=Encoding.JSON) -> Dict[str, Any]:
    if False:
        return 10
    '\n    Decode example from an encoding format.\n\n    Args:\n        example (str): example to decode\n        label_keys (List[str]): list of label keys to check for\n        encoding (Encoding): encoding format to use\n    Returns:\n        Dict[str, Any]: decoded example\n    '
    example = example.strip()
    if encoding == Encoding.JSON:
        return try_decode_json(example, label_keys)
    elif encoding == Encoding.XML:
        raise NotImplementedError('XML encoding not implemented.')
    else:
        raise ValueError(f'Unknown encoding {encoding}.')

def try_decode_json(example: str, label_keys: List[str]) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    'Try to decode an example in a JSON encoding.'
    start = example.find('{')
    end_index = start + 1
    last_error = None
    while -1 < (end_index := example.find('}', end_index + 1)) < len(example):
        try:
            example_dict = dict(json.loads(example[start:end_index + 1]))
            assert any((label_key in example_dict for label_key in label_keys)), f'Failed to decode example.  No label keys found in example: {example_dict}'
            return example_dict
        except Exception as e:
            last_error = e
    if last_error is not None:
        raise last_error
    else:
        raise ValueError('Failed to decode example: ' + example)

def get_majority_value(numbers):
    if False:
        while True:
            i = 10
    logger.info(f'#######################\nGetting majority for {numbers}\n#########################')
    is_dic = any((type(element) is dict for element in numbers))
    if is_dic:
        keys_set = set()
        for item in numbers:
            for key in item:
                keys_set.add(key)
        majority_dic = {}
        for key in keys_set:
            _numbers = []
            for item in numbers:
                if key in item:
                    _numbers.append(item[key])
            maj_val = get_majority_value(_numbers)
            majority_dic[key] = maj_val
        logger.info(f'Majority value is {majority_dic}')
        return majority_dic
    else:
        counter = Counter(numbers)
        (majority_value, _) = counter.most_common(1)[0]
        logger.info(f'Majority value is {majority_value}')
        return majority_value

def try_parse_samples(samples: List[str], prompt_template: PromptTemplate, n_inputs: int, n_samples: int, job_idx: int) -> Tuple[int, List[List[dict]]]:
    if False:
        i = 10
        return i + 15
    'Try to parse a list of samples into a list of examples.\n\n    Args:\n        samples (List[str]): List of samples to parse.\n        prompt_template (PromptTemplate): Prompt template used to generate prompts.\n        n_inputs (int): Number of inputs expected back in the completion.\n        n_samples (int): Number of samples expected back in the completion.\n        job_idx (int): Job index.\n\n    Returns:\n        Tuple[int, List[List[dict]]]: Number of failed samples, and list of examples.\n    '
    output_examples = []
    num_failed = 0
    for (sample_idx, sample) in enumerate(samples):
        try:
            sample_examples = prompt_template.split_output_examples(sample)
            if len(sample_examples) < n_inputs:
                raise ValueError(f'Expected at least {n_inputs} examples, but got {len(sample_examples)}')
            sample_examples = sample_examples[:n_inputs]
        except Exception as e:
            logger.info(f'Failed to split: Job #{job_idx} - sample #{sample_idx + 1}/{n_samples}. Error: {e}')
            output_examples.append(None)
            num_failed += 1
            continue
        example = None
        try:
            sample_examples_parsed = []
            for example in sample_examples:
                sample_examples_parsed.append(decode_example(example, prompt_template.label_keys))
            output_examples.append(sample_examples_parsed)
        except Exception:
            output_examples.append([{} for _ in range(len(sample_examples))])
            num_failed += 1
            logger.exception(f'Failed to decode: Job #{job_idx} - sample #{sample_idx + 1}/{n_samples}')
    return (num_failed, output_examples)

def batch_list(unbatched: list, batch_size: int) -> List[list]:
    if False:
        return 10
    'Batch a list into a list of lists of size batch_size.'
    return [unbatched[i:i + batch_size] for i in range(0, len(unbatched), batch_size)]