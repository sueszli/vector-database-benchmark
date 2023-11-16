from typing import Dict, Any, List, Optional
import json
import pprint
import logging
from collections import defaultdict
import pandas as pd
from haystack.schema import Document, Answer
logger = logging.getLogger(__name__)

def print_answers(results: dict, details: str='all', max_text_len: Optional[int]=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Utility function to print results of Haystack pipelines\n    :param results: Results that the pipeline returned.\n    :param details: Defines the level of details to print. Possible values: minimum, medium, all.\n    :param max_text_len: Specifies the maximum allowed length for a text field. If you don't want to shorten the text, set this value to None.\n    :return: None\n    "
    fields_to_keep_by_level = {'minimum': {Answer: ['answer', 'context']}, 'medium': {Answer: ['answer', 'context', 'score']}}
    if not 'answers' in results.keys():
        raise ValueError(f"The results object does not seem to come from a Reader: it does not contain the 'answers' key, but only: {results.keys()}.  Try print_documents or print_questions.")
    pp = pprint.PrettyPrinter(indent=4)
    queries = []
    if 'query' in results.keys():
        queries = [results['query']]
        answers_lists = [results['answers']]
    elif 'queries' in results.keys():
        queries = results['queries']
        answers_lists = results['answers']
    for (query_idx, answers) in enumerate(answers_lists):
        filtered_answers = []
        if details in fields_to_keep_by_level.keys():
            for ans in answers:
                filtered_ans = {field: getattr(ans, field) for field in fields_to_keep_by_level[details][type(ans)] if getattr(ans, field) is not None}
                filtered_answers.append(filtered_ans)
        elif details == 'all':
            filtered_answers = answers
        else:
            valid_values = ', '.join(fields_to_keep_by_level.keys()) + " and 'all'"
            logging.warn("print_answers received details='%s', which was not understood. ", details)
            logging.warn("Valid values are %s. Using 'all'.", valid_values)
            filtered_answers = answers
        if max_text_len is not None:
            for ans in answers:
                if getattr(ans, 'context') and len(ans.context) > max_text_len:
                    ans.context = ans.context[:max_text_len] + '...'
        if len(queries) > 0:
            pp.pprint(f'Query: {queries[query_idx]}')
        pp.pprint('Answers:')
        pp.pprint(filtered_answers)

def print_documents(results: dict, max_text_len: Optional[int]=None, print_name: bool=True, print_meta: bool=False):
    if False:
        print('Hello World!')
    "\n    Utility that prints a compressed representation of the documents returned by a pipeline.\n    :param max_text_len: Shorten the document's content to a maximum number of characters. When set to `None`, the document is not shortened.\n    :param print_name: Whether to print the document's name from the metadata.\n    :param print_meta: Whether to print the document's metadata.\n    "
    print(f"\nQuery: {results['query']}\n")
    pp = pprint.PrettyPrinter(indent=4)
    if any((not isinstance(doc, Document) for doc in results['documents'])):
        raise ValueError("This results object does not contain `Document` objects under the `documents` key. Please make sure the last node of your pipeline makes proper use of the new Haystack primitive objects, and if you're using Haystack nodes/pipelines only, please report this as a bug.")
    for doc in results['documents']:
        content = doc.content
        if max_text_len:
            content = doc.content[:max_text_len] + ('...' if len(doc.content) > max_text_len else '')
        results = {'content': content}
        if print_name:
            results['name'] = doc.meta.get('name', None)
        if print_meta:
            results['meta'] = doc.meta
        pp.pprint(results)
        print()

def print_questions(results: dict):
    if False:
        return 10
    '\n    Utility to print the output of a question generating pipeline in a readable format.\n    '
    if 'generated_questions' in results.keys():
        print('\nGenerated questions:')
        for result in results['generated_questions']:
            for question in result['questions']:
                print(f' - {question}')
    elif 'queries' in results.keys() and 'answers' in results.keys():
        print('\nGenerated pairs:')
        for (query, answers) in zip(results['queries'], results['answers']):
            print(f' - Q: {query}')
            for answer in answers:
                if not isinstance(answer, Answer):
                    raise ValueError("This results object does not contain `Answer` objects under the `answers` key of the generated question/answer pairs. Please make sure the last node of your pipeline makes proper use of the new Haystack primitive objects, and if you're using Haystack nodes/pipelines only, please report this as a bug.")
                print(f'      A: {answer.answer}')
    else:
        raise ValueError(f"This object does not seem to be the output of a question generating pipeline: does not contain neither 'generated_questions' nor 'results', but only: {results.keys()}.  Try `print_answers` or `print_documents`.")

def export_answers_to_csv(agg_results: list, output_file):
    if False:
        print('Hello World!')
    '\n    Exports answers coming from finder.get_answers() to a CSV file.\n    :param agg_results: A list of predictions coming from finder.get_answers().\n    :param output_file: The name of the output file.\n    :return: None\n    '
    if isinstance(agg_results, dict):
        agg_results = [agg_results]
    assert 'query' in agg_results[0], f'Wrong format used for {agg_results[0]}'
    assert 'answers' in agg_results[0], f'Wrong format used for {agg_results[0]}'
    data = {}
    data['query'] = []
    data['prediction'] = []
    data['prediction_rank'] = []
    data['prediction_context'] = []
    for res in agg_results:
        for i in range(len(res['answers'])):
            temp = res['answers'][i]
            data['query'].append(res['query'])
            data['prediction'].append(temp.answer)
            data['prediction_rank'].append(i + 1)
            data['prediction_context'].append(temp.context)
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

def convert_labels_to_squad(labels_file: str):
    if False:
        i = 10
        return i + 15
    '\n    Convert the export from the labeling UI to the SQuAD format for training.\n\n    :param labels_file: The path to the file containing labels.\n    :return:\n    '
    from haystack.document_stores.sql import DocumentORM
    with open(labels_file, encoding='utf-8') as label_file:
        labels = json.load(label_file)
    labels_grouped_by_documents = defaultdict(list)
    for label in labels:
        labels_grouped_by_documents[label['document_id']].append(label)
    labels_in_squad_format = {'data': []}
    for (document_id, labels) in labels_grouped_by_documents.items():
        qas = []
        for label in labels:
            doc = DocumentORM.query.get(label['document_id'])
            assert doc.content[label['start_offset']:label['end_offset']] == label['selected_text']
            qas.append({'question': label['question'], 'id': label['id'], 'question_id': label['question_id'], 'answers': [{'text': label['selected_text'], 'answer_start': label['start_offset'], 'labeller_id': label['labeler_id']}], 'is_impossible': False})
        squad_format_label = {'paragraphs': [{'qas': qas, 'context': doc.content, 'document_id': document_id}]}
        labels_in_squad_format['data'].append(squad_format_label)
    with open('labels_in_squad_format.json', 'w+', encoding='utf-8') as outfile:
        json.dump(labels_in_squad_format, outfile)