"""Functions for importing/exporting Object Detection categories."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import tensorflow as tf

def load_categories_from_csv_file(csv_path):
    if False:
        i = 10
        return i + 15
    'Loads categories from a csv file.\n\n  The CSV file should have one comma delimited numeric category id and string\n  category name pair per line. For example:\n\n  0,"cat"\n  1,"dog"\n  2,"bird"\n  ...\n\n  Args:\n    csv_path: Path to the csv file to be parsed into categories.\n  Returns:\n    categories: A list of dictionaries representing all possible categories.\n                The categories will contain an integer \'id\' field and a string\n                \'name\' field.\n  Raises:\n    ValueError: If the csv file is incorrectly formatted.\n  '
    categories = []
    with tf.gfile.Open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            if not row:
                continue
            if len(row) != 2:
                raise ValueError('Expected 2 fields per row in csv: %s' % ','.join(row))
            category_id = int(row[0])
            category_name = row[1]
            categories.append({'id': category_id, 'name': category_name})
    return categories

def save_categories_to_csv_file(categories, csv_path):
    if False:
        return 10
    "Saves categories to a csv file.\n\n  Args:\n    categories: A list of dictionaries representing categories to save to file.\n                Each category must contain an 'id' and 'name' field.\n    csv_path: Path to the csv file to be parsed into categories.\n  "
    categories.sort(key=lambda x: x['id'])
    with tf.gfile.Open(csv_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"')
        for category in categories:
            writer.writerow([category['id'], category['name']])