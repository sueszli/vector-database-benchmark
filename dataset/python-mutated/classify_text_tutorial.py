"""Using the classify_text method to find content categories of text files,
Then use the content category labels to compare text similarity.

For more information, see the tutorial page at
https://cloud.google.com/natural-language/docs/classify-text-tutorial.
"""
import argparse
import json
import os
from google.cloud import language_v1
import numpy

def classify(text, verbose=True):
    if False:
        for i in range(10):
            print('nop')
    'Classify the input text into categories.'
    language_client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = language_client.classify_text(request={'document': document})
    categories = response.categories
    result = {}
    for category in categories:
        result[category.name] = category.confidence
    if verbose:
        print(text)
        for category in categories:
            print('=' * 20)
            print('{:<16}: {}'.format('category', category.name))
            print('{:<16}: {}'.format('confidence', category.confidence))
    return result

def index(path, index_file):
    if False:
        return 10
    'Classify each text file in a directory and write\n    the results to the index_file.\n    '
    result = {}
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if not os.path.isfile(file_path):
            continue
        try:
            with open(file_path) as f:
                text = f.read()
                categories = classify(text, verbose=False)
                result[filename] = categories
        except Exception:
            print(f'Failed to process {file_path}')
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False))
    print(f'Texts indexed in file: {index_file}')
    return result

def split_labels(categories):
    if False:
        for i in range(10):
            print('nop')
    'The category labels are of the form "/a/b/c" up to three levels,\n    for example "/Computers & Electronics/Software", and these labels\n    are used as keys in the categories dictionary, whose values are\n    confidence scores.\n\n    The split_labels function splits the keys into individual levels\n    while duplicating the confidence score, which allows a natural\n    boost in how we calculate similarity when more levels are in common.\n\n    Example:\n    If we have\n\n    x = {"/a/b/c": 0.5}\n    y = {"/a/b": 0.5}\n    z = {"/a": 0.5}\n\n    Then x and y are considered more similar than y and z.\n    '
    _categories = {}
    for (name, confidence) in categories.items():
        labels = [label for label in name.split('/') if label]
        for label in labels:
            _categories[label] = confidence
    return _categories

def similarity(categories1, categories2):
    if False:
        i = 10
        return i + 15
    'Cosine similarity of the categories treated as sparse vectors.'
    categories1 = split_labels(categories1)
    categories2 = split_labels(categories2)
    norm1 = numpy.linalg.norm(list(categories1.values()))
    norm2 = numpy.linalg.norm(list(categories2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    dot = 0.0
    for (label, confidence) in categories1.items():
        dot += confidence * categories2.get(label, 0.0)
    return dot / (norm1 * norm2)

def query(index_file, text, n_top=3):
    if False:
        i = 10
        return i + 15
    'Find the indexed files that are the most similar to\n    the query text.\n    '
    with open(index_file) as f:
        index = json.load(f)
    query_categories = classify(text, verbose=False)
    similarities = []
    for (filename, categories) in index.items():
        similarities.append((filename, similarity(query_categories, categories)))
    similarities = sorted(similarities, key=lambda p: p[1], reverse=True)
    print('=' * 20)
    print(f'Query: {text}\n')
    for (category, confidence) in query_categories.items():
        print(f'\tCategory: {category}, confidence: {confidence}')
    print(f'\nMost similar {n_top} indexed texts:')
    for (filename, sim) in similarities[:n_top]:
        print(f'\tFilename: {filename}')
        print(f'\tSimilarity: {sim}')
        print('\n')
    return similarities

def query_category(index_file, category_string, n_top=3):
    if False:
        return 10
    'Find the indexed files that are the most similar to\n    the query label.\n\n    The list of all available labels:\n    https://cloud.google.com/natural-language/docs/categories\n    '
    with open(index_file) as f:
        index = json.load(f)
    query_categories = {category_string: 1.0}
    similarities = []
    for (filename, categories) in index.items():
        similarities.append((filename, similarity(query_categories, categories)))
    similarities = sorted(similarities, key=lambda p: p[1], reverse=True)
    print('=' * 20)
    print(f'Query: {category_string}\n')
    print(f'\nMost similar {n_top} indexed texts:')
    for (filename, sim) in similarities[:n_top]:
        print(f'\tFilename: {filename}')
        print(f'\tSimilarity: {sim}')
        print('\n')
    return similarities
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    classify_parser = subparsers.add_parser('classify', help=classify.__doc__)
    classify_parser.add_argument('text', help='The text to be classified. The text needs to have at least 20 tokens.')
    index_parser = subparsers.add_parser('index', help=index.__doc__)
    index_parser.add_argument('path', help='The directory that contains text files to be indexed.')
    index_parser.add_argument('--index_file', help='Filename for the output JSON.', default='index.json')
    query_parser = subparsers.add_parser('query', help=query.__doc__)
    query_parser.add_argument('index_file', help='Path to the index JSON file.')
    query_parser.add_argument('text', help='Query text.')
    query_category_parser = subparsers.add_parser('query-category', help=query_category.__doc__)
    query_category_parser.add_argument('index_file', help='Path to the index JSON file.')
    query_category_parser.add_argument('category', help='Query category.')
    args = parser.parse_args()
    if args.command == 'classify':
        classify(args.text)
    if args.command == 'index':
        index(args.path, args.index_file)
    if args.command == 'query':
        query(args.index_file, args.text)
    if args.command == 'query-category':
        query_category(args.index_file, args.category)