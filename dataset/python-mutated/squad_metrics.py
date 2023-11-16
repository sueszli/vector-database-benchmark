"""
Very heavily inspired by the official evaluation script for SQuAD version 2.0 which was modified by XLNet authors to
update `find_best_threshold` scripts for SQuAD V2.0

In addition to basic functionality, we also compute additional statistics and plot precision-recall curves if an
additional na_prob.json file is provided. This file is expected to map question ID's to the model's predicted
probability that a question is unanswerable.
"""
import collections
import json
import math
import re
import string
from ...models.bert import BasicTokenizer
from ...utils import logging
logger = logging.get_logger(__name__)

def normalize_answer(s):
    if False:
        print('Hello World!')
    'Lower text and remove punctuation, articles and extra whitespace.'

    def remove_articles(text):
        if False:
            while True:
                i = 10
        regex = re.compile('\\b(a|an|the)\\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        if False:
            for i in range(10):
                print('nop')
        return ' '.join(text.split())

    def remove_punc(text):
        if False:
            i = 10
            return i + 15
        exclude = set(string.punctuation)
        return ''.join((ch for ch in text if ch not in exclude))

    def lower(text):
        if False:
            for i in range(10):
                print('nop')
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if False:
        while True:
            i = 10
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    if False:
        for i in range(10):
            print('nop')
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    if False:
        return 10
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def get_raw_scores(examples, preds):
    if False:
        for i in range(10):
            print('nop')
    '\n    Computes the exact and f1 scores from the examples and the model predictions\n    '
    exact_scores = {}
    f1_scores = {}
    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer['text'] for answer in example.answers if normalize_answer(answer['text'])]
        if not gold_answers:
            gold_answers = ['']
        if qas_id not in preds:
            print(f'Missing prediction for {qas_id}')
            continue
        prediction = preds[qas_id]
        exact_scores[qas_id] = max((compute_exact(a, prediction) for a in gold_answers))
        f1_scores[qas_id] = max((compute_f1(a, prediction) for a in gold_answers))
    return (exact_scores, f1_scores)

def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    if False:
        return 10
    new_scores = {}
    for (qid, s) in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores

def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if False:
        i = 10
        return i + 15
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([('exact', 100.0 * sum(exact_scores.values()) / total), ('f1', 100.0 * sum(f1_scores.values()) / total), ('total', total)])
    else:
        total = len(qid_list)
        return collections.OrderedDict([('exact', 100.0 * sum((exact_scores[k] for k in qid_list)) / total), ('f1', 100.0 * sum((f1_scores[k] for k in qid_list)) / total), ('total', total)])

def merge_eval(main_eval, new_eval, prefix):
    if False:
        print('Hello World!')
    for k in new_eval:
        main_eval[f'{prefix}_{k}'] = new_eval[k]

def find_best_thresh_v2(preds, scores, na_probs, qid_to_has_ans):
    if False:
        while True:
            i = 10
    num_no_ans = sum((1 for k in qid_to_has_ans if not qid_to_has_ans[k]))
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for (i, qid) in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        elif preds[qid]:
            diff = -1
        else:
            diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    (has_ans_score, has_ans_cnt) = (0, 0)
    for qid in qid_list:
        if not qid_to_has_ans[qid]:
            continue
        has_ans_cnt += 1
        if qid not in scores:
            continue
        has_ans_score += scores[qid]
    return (100.0 * best_score / len(scores), best_thresh, 1.0 * has_ans_score / has_ans_cnt)

def find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    if False:
        return 10
    (best_exact, exact_thresh, has_ans_exact) = find_best_thresh_v2(preds, exact_raw, na_probs, qid_to_has_ans)
    (best_f1, f1_thresh, has_ans_f1) = find_best_thresh_v2(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval['best_exact'] = best_exact
    main_eval['best_exact_thresh'] = exact_thresh
    main_eval['best_f1'] = best_f1
    main_eval['best_f1_thresh'] = f1_thresh
    main_eval['has_ans_exact'] = has_ans_exact
    main_eval['has_ans_f1'] = has_ans_f1

def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    if False:
        for i in range(10):
            print('nop')
    num_no_ans = sum((1 for k in qid_to_has_ans if not qid_to_has_ans[k]))
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for (_, qid) in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        elif preds[qid]:
            diff = -1
        else:
            diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return (100.0 * best_score / len(scores), best_thresh)

def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    if False:
        print('Hello World!')
    (best_exact, exact_thresh) = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    (best_f1, f1_thresh) = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval['best_exact'] = best_exact
    main_eval['best_exact_thresh'] = exact_thresh
    main_eval['best_f1'] = best_f1
    main_eval['best_f1_thresh'] = f1_thresh

def squad_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
    if False:
        return 10
    qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    has_answer_qids = [qas_id for (qas_id, has_answer) in qas_id_to_has_answer.items() if has_answer]
    no_answer_qids = [qas_id for (qas_id, has_answer) in qas_id_to_has_answer.items() if not has_answer]
    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}
    (exact, f1) = get_raw_scores(examples, preds)
    exact_threshold = apply_no_ans_threshold(exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold)
    f1_threshold = apply_no_ans_threshold(f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold)
    evaluation = make_eval_dict(exact_threshold, f1_threshold)
    if has_answer_qids:
        has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=has_answer_qids)
        merge_eval(evaluation, has_ans_eval, 'HasAns')
    if no_answer_qids:
        no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=no_answer_qids)
        merge_eval(evaluation, no_ans_eval, 'NoAns')
    if no_answer_probs:
        find_all_best_thresh(evaluation, preds, exact, f1, no_answer_probs, qas_id_to_has_answer)
    return evaluation

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    if False:
        print('Hello World!')
    'Project the tokenized prediction back to the original text.'

    def _strip_spaces(text):
        if False:
            while True:
                i = 10
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == ' ':
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = ''.join(ns_chars)
        return (ns_text, ns_to_s_map)
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    tok_text = ' '.join(tokenizer.tokenize(orig_text))
    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(f"Unable to find text: '{pred_text}' in '{orig_text}'")
        return orig_text
    end_position = start_position + len(pred_text) - 1
    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)
    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info(f"Length not equal after stripping spaces: '{orig_ns_text}' vs '{tok_ns_text}'")
        return orig_text
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i
    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]
    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text
    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]
    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text
    output_text = orig_text[orig_start_position:orig_end_position + 1]
    return output_text

def _get_best_indexes(logits, n_best_size):
    if False:
        print('Hello World!')
    'Get the n-best logits from a list.'
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def _compute_softmax(scores):
    if False:
        print('Hello World!')
    'Compute softmax probability over raw logits.'
    if not scores:
        return []
    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score
    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x
    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def compute_predictions_logits(all_examples, all_features, all_results, n_best_size, max_answer_length, do_lower_case, output_prediction_file, output_nbest_file, output_null_log_odds_file, verbose_logging, version_2_with_negative, null_score_diff_threshold, tokenizer):
    if False:
        print('Hello World!')
    'Write final predictions to the json file and log-odds of null if needed.'
    if output_prediction_file:
        logger.info(f'Writing predictions to: {output_prediction_file}')
    if output_nbest_file:
        logger.info(f'Writing nbest to: {output_nbest_file}')
    if output_null_log_odds_file and version_2_with_negative:
        logger.info(f'Writing null_log_odds to: {output_null_log_odds_file}')
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
    _PrelimPrediction = collections.namedtuple('PrelimPrediction', ['feature_index', 'start_index', 'end_index', 'start_logit', 'end_logit'])
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        score_null = 1000000
        min_null_feature_index = 0
        null_start_logit = 0
        null_end_logit = 0
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(_PrelimPrediction(feature_index=feature_index, start_index=start_index, end_index=end_index, start_logit=result.start_logits[start_index], end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(_PrelimPrediction(feature_index=min_null_feature_index, start_index=0, end_index=0, start_logit=null_start_logit, end_logit=null_end_logit))
        prelim_predictions = sorted(prelim_predictions, key=lambda x: x.start_logit + x.end_logit, reverse=True)
        _NbestPrediction = collections.namedtuple('NbestPrediction', ['text', 'start_logit', 'end_logit'])
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:
                tok_tokens = feature.tokens[pred.start_index:pred.end_index + 1]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:orig_doc_end + 1]
                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)
                tok_text = tok_text.strip()
                tok_text = ' '.join(tok_text.split())
                orig_text = ' '.join(orig_tokens)
                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
            else:
                final_text = ''
                seen_predictions[final_text] = True
            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))
        if version_2_with_negative:
            if '' not in seen_predictions:
                nbest.append(_NbestPrediction(text='', start_logit=null_start_logit, end_logit=null_end_logit))
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text='empty', start_logit=0.0, end_logit=0.0))
        if not nbest:
            nbest.append(_NbestPrediction(text='empty', start_logit=0.0, end_logit=0.0))
        if len(nbest) < 1:
            raise ValueError('No valid predictions')
        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry
        probs = _compute_softmax(total_scores)
        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output['text'] = entry.text
            output['probability'] = probs[i]
            output['start_logit'] = entry.start_logit
            output['end_logit'] = entry.end_logit
            nbest_json.append(output)
        if len(nbest_json) < 1:
            raise ValueError('No valid predictions')
        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]['text']
        else:
            score_diff = score_null - best_non_null_entry.start_logit - best_non_null_entry.end_logit
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ''
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json
    if output_prediction_file:
        with open(output_prediction_file, 'w') as writer:
            writer.write(json.dumps(all_predictions, indent=4) + '\n')
    if output_nbest_file:
        with open(output_nbest_file, 'w') as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + '\n')
    if output_null_log_odds_file and version_2_with_negative:
        with open(output_null_log_odds_file, 'w') as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + '\n')
    return all_predictions

def compute_predictions_log_probs(all_examples, all_features, all_results, n_best_size, max_answer_length, output_prediction_file, output_nbest_file, output_null_log_odds_file, start_n_top, end_n_top, version_2_with_negative, tokenizer, verbose_logging):
    if False:
        i = 10
        return i + 15
    "\n    XLNet write prediction logic (more complex than Bert's). Write final predictions to the json file and log-odds of\n    null if needed.\n\n    Requires utils_squad_evaluate.py\n    "
    _PrelimPrediction = collections.namedtuple('PrelimPrediction', ['feature_index', 'start_index', 'end_index', 'start_log_prob', 'end_log_prob'])
    _NbestPrediction = collections.namedtuple('NbestPrediction', ['text', 'start_log_prob', 'end_log_prob'])
    logger.info(f'Writing predictions to: {output_prediction_file}')
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        score_null = 1000000
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            cur_null_score = result.cls_logits
            score_null = min(score_null, cur_null_score)
            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_logits[i]
                    start_index = result.start_top_index[i]
                    j_index = i * end_n_top + j
                    end_log_prob = result.end_logits[j_index]
                    end_index = result.end_top_index[j_index]
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(_PrelimPrediction(feature_index=feature_index, start_index=start_index, end_index=end_index, start_log_prob=start_log_prob, end_log_prob=end_log_prob))
        prelim_predictions = sorted(prelim_predictions, key=lambda x: x.start_log_prob + x.end_log_prob, reverse=True)
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            tok_tokens = feature.tokens[pred.start_index:pred.end_index + 1]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:orig_doc_end + 1]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)
            tok_text = tok_text.strip()
            tok_text = ' '.join(tok_text.split())
            orig_text = ' '.join(orig_tokens)
            if hasattr(tokenizer, 'do_lower_case'):
                do_lower_case = tokenizer.do_lower_case
            else:
                do_lower_case = tokenizer.do_lowercase_and_remove_accent
            final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
            if final_text in seen_predictions:
                continue
            seen_predictions[final_text] = True
            nbest.append(_NbestPrediction(text=final_text, start_log_prob=pred.start_log_prob, end_log_prob=pred.end_log_prob))
        if not nbest:
            nbest.append(_NbestPrediction(text='', start_log_prob=-1000000.0, end_log_prob=-1000000.0))
        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry
        probs = _compute_softmax(total_scores)
        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output['text'] = entry.text
            output['probability'] = probs[i]
            output['start_log_prob'] = entry.start_log_prob
            output['end_log_prob'] = entry.end_log_prob
            nbest_json.append(output)
        if len(nbest_json) < 1:
            raise ValueError('No valid predictions')
        if best_non_null_entry is None:
            raise ValueError('No valid predictions')
        score_diff = score_null
        scores_diff_json[example.qas_id] = score_diff
        all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json
    with open(output_prediction_file, 'w') as writer:
        writer.write(json.dumps(all_predictions, indent=4) + '\n')
    with open(output_nbest_file, 'w') as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + '\n')
    if version_2_with_negative:
        with open(output_null_log_odds_file, 'w') as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + '\n')
    return all_predictions