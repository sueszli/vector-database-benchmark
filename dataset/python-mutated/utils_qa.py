"""
Post-processing utilities for question answering.
"""
import collections
import json
import logging
import os
from typing import Optional, Tuple
import numpy as np
from tqdm.auto import tqdm
logger = logging.getLogger(__name__)

def postprocess_qa_predictions(examples, features, predictions: Tuple[np.ndarray, np.ndarray], version_2_with_negative: bool=False, n_best_size: int=20, max_answer_length: int=30, null_score_diff_threshold: float=0.0, output_dir: Optional[str]=None, prefix: Optional[str]=None, log_level: Optional[int]=logging.WARNING):
    if False:
        for i in range(10):
            print('nop')
    '\n    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the\n    original contexts. This is the base postprocessing functions for models that only return start and end logits.\n\n    Args:\n        examples: The non-preprocessed dataset (see the main script for more information).\n        features: The processed dataset (see the main script for more information).\n        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):\n            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its\n            first dimension must match the number of elements of :obj:`features`.\n        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):\n            Whether or not the underlying dataset contains examples with no answers.\n        n_best_size (:obj:`int`, `optional`, defaults to 20):\n            The total number of n-best predictions to generate when looking for an answer.\n        max_answer_length (:obj:`int`, `optional`, defaults to 30):\n            The maximum length of an answer that can be generated. This is needed because the start and end predictions\n            are not conditioned on one another.\n        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):\n            The threshold used to select the null answer: if the best answer has a score that is less than the score of\n            the null answer minus this threshold, the null answer is selected for this example (note that the score of\n            the null answer for an example giving several features is the minimum of the scores for the null answer on\n            each feature: all features must be aligned on the fact they `want` to predict a null answer).\n\n            Only useful when :obj:`version_2_with_negative` is :obj:`True`.\n        output_dir (:obj:`str`, `optional`):\n            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if\n            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null\n            answers, are saved in `output_dir`.\n        prefix (:obj:`str`, `optional`):\n            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.\n        log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):\n            ``logging`` log level (e.g., ``logging.WARNING``)\n    '
    if len(predictions) != 2:
        raise ValueError('`predictions` should be a tuple with two elements (start_logits, end_logits).')
    (all_start_logits, all_end_logits) = predictions
    if len(predictions[0]) != len(features):
        raise ValueError(f'Got {len(predictions[0])} predictions and {len(features)} features.')
    example_id_to_index = {k: i for (i, k) in enumerate(examples['id'])}
    features_per_example = collections.defaultdict(list)
    for (i, feature) in enumerate(features):
        features_per_example[example_id_to_index[feature['example_id']]].append(i)
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()
    logger.setLevel(log_level)
    logger.info(f'Post-processing {len(examples)} example predictions split into {len(features)} features.')
    for (example_index, example) in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]
        min_null_prediction = None
        prelim_predictions = []
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]['offset_mapping']
            token_is_max_context = features[feature_index].get('token_is_max_context', None)
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction['score'] > feature_null_score:
                min_null_prediction = {'offsets': (0, 0), 'score': feature_null_score, 'start_logit': start_logits[0], 'end_logit': end_logits[0]}
            start_indexes = np.argsort(start_logits)[-1:-n_best_size - 1:-1].tolist()
            end_indexes = np.argsort(end_logits)[-1:-n_best_size - 1:-1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping) or offset_mapping[start_index] is None or (len(offset_mapping[start_index]) < 2) or (offset_mapping[end_index] is None) or (len(offset_mapping[end_index]) < 2):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    if token_is_max_context is not None and (not token_is_max_context.get(str(start_index), False)):
                        continue
                    prelim_predictions.append({'offsets': (offset_mapping[start_index][0], offset_mapping[end_index][1]), 'score': start_logits[start_index] + end_logits[end_index], 'start_logit': start_logits[start_index], 'end_logit': end_logits[end_index]})
        if version_2_with_negative and min_null_prediction is not None:
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction['score']
        predictions = sorted(prelim_predictions, key=lambda x: x['score'], reverse=True)[:n_best_size]
        if version_2_with_negative and min_null_prediction is not None and (not any((p['offsets'] == (0, 0) for p in predictions))):
            predictions.append(min_null_prediction)
        context = example['context']
        for pred in predictions:
            offsets = pred.pop('offsets')
            pred['text'] = context[offsets[0]:offsets[1]]
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]['text'] == ''):
            predictions.insert(0, {'text': 'empty', 'start_logit': 0.0, 'end_logit': 0.0, 'score': 0.0})
        scores = np.array([pred.pop('score') for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        for (prob, pred) in zip(probs, predictions):
            pred['probability'] = prob
        if not version_2_with_negative:
            all_predictions[example['id']] = predictions[0]['text']
        else:
            i = 0
            while predictions[i]['text'] == '':
                i += 1
            best_non_null_pred = predictions[i]
            score_diff = null_score - best_non_null_pred['start_logit'] - best_non_null_pred['end_logit']
            scores_diff_json[example['id']] = float(score_diff)
            if score_diff > null_score_diff_threshold:
                all_predictions[example['id']] = ''
            else:
                all_predictions[example['id']] = best_non_null_pred['text']
        all_nbest_json[example['id']] = [{k: float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v for (k, v) in pred.items()} for pred in predictions]
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f'{output_dir} is not a directory.')
        prediction_file = os.path.join(output_dir, 'predictions.json' if prefix is None else f'{prefix}_predictions.json')
        nbest_file = os.path.join(output_dir, 'nbest_predictions.json' if prefix is None else f'{prefix}_nbest_predictions.json')
        if version_2_with_negative:
            null_odds_file = os.path.join(output_dir, 'null_odds.json' if prefix is None else f'{prefix}_null_odds.json')
        logger.info(f'Saving predictions to {prediction_file}.')
        with open(prediction_file, 'w') as writer:
            writer.write(json.dumps(all_predictions, indent=4) + '\n')
        logger.info(f'Saving nbest_preds to {nbest_file}.')
        with open(nbest_file, 'w') as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + '\n')
        if version_2_with_negative:
            logger.info(f'Saving null_odds to {null_odds_file}.')
            with open(null_odds_file, 'w') as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + '\n')
    return all_predictions

def postprocess_qa_predictions_with_beam_search(examples, features, predictions: Tuple[np.ndarray, np.ndarray], version_2_with_negative: bool=False, n_best_size: int=20, max_answer_length: int=30, start_n_top: int=5, end_n_top: int=5, output_dir: Optional[str]=None, prefix: Optional[str]=None, log_level: Optional[int]=logging.WARNING):
    if False:
        print('Hello World!')
    '\n    Post-processes the predictions of a question-answering model with beam search to convert them to answers that are substrings of the\n    original contexts. This is the postprocessing functions for models that return start and end logits, indices, as well as\n    cls token predictions.\n\n    Args:\n        examples: The non-preprocessed dataset (see the main script for more information).\n        features: The processed dataset (see the main script for more information).\n        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):\n            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its\n            first dimension must match the number of elements of :obj:`features`.\n        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):\n            Whether or not the underlying dataset contains examples with no answers.\n        n_best_size (:obj:`int`, `optional`, defaults to 20):\n            The total number of n-best predictions to generate when looking for an answer.\n        max_answer_length (:obj:`int`, `optional`, defaults to 30):\n            The maximum length of an answer that can be generated. This is needed because the start and end predictions\n            are not conditioned on one another.\n        start_n_top (:obj:`int`, `optional`, defaults to 5):\n            The number of top start logits too keep when searching for the :obj:`n_best_size` predictions.\n        end_n_top (:obj:`int`, `optional`, defaults to 5):\n            The number of top end logits too keep when searching for the :obj:`n_best_size` predictions.\n        output_dir (:obj:`str`, `optional`):\n            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if\n            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null\n            answers, are saved in `output_dir`.\n        prefix (:obj:`str`, `optional`):\n            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.\n        log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):\n            ``logging`` log level (e.g., ``logging.WARNING``)\n    '
    if len(predictions) != 5:
        raise ValueError('`predictions` should be a tuple with five elements.')
    (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits) = predictions
    if len(predictions[0]) != len(features):
        raise ValueError(f'Got {len(predictions[0])} predictions and {len(features)} features.')
    example_id_to_index = {k: i for (i, k) in enumerate(examples['id'])}
    features_per_example = collections.defaultdict(list)
    for (i, feature) in enumerate(features):
        features_per_example[example_id_to_index[feature['example_id']]].append(i)
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict() if version_2_with_negative else None
    logger.setLevel(log_level)
    logger.info(f'Post-processing {len(examples)} example predictions split into {len(features)} features.')
    for (example_index, example) in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        prelim_predictions = []
        for feature_index in feature_indices:
            start_log_prob = start_top_log_probs[feature_index]
            start_indexes = start_top_index[feature_index]
            end_log_prob = end_top_log_probs[feature_index]
            end_indexes = end_top_index[feature_index]
            feature_null_score = cls_logits[feature_index]
            offset_mapping = features[feature_index]['offset_mapping']
            token_is_max_context = features[feature_index].get('token_is_max_context', None)
            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score
            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_index = int(start_indexes[i])
                    j_index = i * end_n_top + j
                    end_index = int(end_indexes[j_index])
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping) or offset_mapping[start_index] is None or (len(offset_mapping[start_index]) < 2) or (offset_mapping[end_index] is None) or (len(offset_mapping[end_index]) < 2):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    if token_is_max_context is not None and (not token_is_max_context.get(str(start_index), False)):
                        continue
                    prelim_predictions.append({'offsets': (offset_mapping[start_index][0], offset_mapping[end_index][1]), 'score': start_log_prob[i] + end_log_prob[j_index], 'start_log_prob': start_log_prob[i], 'end_log_prob': end_log_prob[j_index]})
        predictions = sorted(prelim_predictions, key=lambda x: x['score'], reverse=True)[:n_best_size]
        context = example['context']
        for pred in predictions:
            offsets = pred.pop('offsets')
            pred['text'] = context[offsets[0]:offsets[1]]
        if len(predictions) == 0:
            min_null_score = -2e-06
            predictions.insert(0, {'text': '', 'start_logit': -1e-06, 'end_logit': -1e-06, 'score': min_null_score})
        scores = np.array([pred.pop('score') for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        for (prob, pred) in zip(probs, predictions):
            pred['probability'] = prob
        all_predictions[example['id']] = predictions[0]['text']
        if version_2_with_negative:
            scores_diff_json[example['id']] = float(min_null_score)
        all_nbest_json[example['id']] = [{k: float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v for (k, v) in pred.items()} for pred in predictions]
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f'{output_dir} is not a directory.')
        prediction_file = os.path.join(output_dir, 'predictions.json' if prefix is None else f'{prefix}_predictions.json')
        nbest_file = os.path.join(output_dir, 'nbest_predictions.json' if prefix is None else f'{prefix}_nbest_predictions.json')
        if version_2_with_negative:
            null_odds_file = os.path.join(output_dir, 'null_odds.json' if prefix is None else f'{prefix}_null_odds.json')
        logger.info(f'Saving predictions to {prediction_file}.')
        with open(prediction_file, 'w') as writer:
            writer.write(json.dumps(all_predictions, indent=4) + '\n')
        logger.info(f'Saving nbest_preds to {nbest_file}.')
        with open(nbest_file, 'w') as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + '\n')
        if version_2_with_negative:
            logger.info(f'Saving null_odds to {null_odds_file}.')
            with open(null_odds_file, 'w') as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + '\n')
    return (all_predictions, scores_diff_json)