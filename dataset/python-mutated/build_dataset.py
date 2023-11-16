import argparse
import csv
import json as json
import os
import re
import spacy
import numpy as np
from collections import Counter
from utils import most_common_fonts, font_ratios, markdown_to_text
nlp = spacy.load('en_core_web_sm')

def walk_line(filename_line, page_number, node_line, acc_line, common_fonts, font_ratios):
    if False:
        i = 10
        return i + 15
    line = ''
    fonts_ids = []
    is_title_case = True
    for word in node_line['content']:
        text = word['content']
        if len(text) > 4:
            is_title_case = is_title_case and (bool(re.match('^[A-Z]\\w+', text)) or bool(re.match('^(?:\\W*\\d+\\W*)+\\w+', text)))
        fonts_ids.append(word['font'])
        line += text + ' '
    line_font = filename_line['fonts'][Counter(fonts_ids).most_common(1)[0][0] - 1]
    is_bold = all((filename_line['fonts'][font_id - 1]['weight'] == 'bold' for font_id in fonts_ids))
    line_font_ratio = font_ratios[str(line_font['id']) + '_' + str(page_number)]
    if line.islower():
        text_case = 0
    elif line.isupper():
        text_case = 1
    elif is_title_case:
        text_case = 2
    else:
        text_case = 3
    doc = nlp(line)
    nb_verbs = len([token.lemma_ for token in doc if token.pos_ == 'VERB'])
    nb_nouns = len([chunk.text for chunk in doc.noun_chunks])
    nb_cardinal = len([entity.text for entity in doc.ents if entity.label_ == 'CARDINAL'])
    is_different_style = int(all([is_bold ^ (common_fonts[i]['weight'] == 'bold') for i in range(len(common_fonts))]))
    is_font_bigger = int(all([line_font['size'] > common_fonts[i]['size'] for i in range(len(common_fonts))]))
    different_color = int(all([line_font['color'] != common_fonts[i]['color'] for i in range(len(common_fonts))]))
    acc_line.append([line.strip(), is_different_style, is_font_bigger, different_color, int(len(set(fonts_ids)) == 1), text_case, len(node_line['content']), int(bool(re.match('^\\d*\\.?\\d*$', line.strip()))), nb_verbs, nb_nouns, nb_cardinal, line_font['size'], int(is_bold), line_font_ratio, 0, 'paragraph', False])

def walk(filename, page_number, node, acc, common_fonts, font_ratios):
    if False:
        print('Hello World!')
    elements_to_consider = {'paragraph', 'heading', 'list'}
    if node['type'] == 'line':
        walk_line(filename, page_number, node, acc, common_fonts, font_ratios)
    elif node['type'] in elements_to_consider:
        for elem in node['content']:
            walk(filename, page_number, elem, acc, common_fonts, font_ratios)

def extract_lines(file):
    if False:
        i = 10
        return i + 15
    lines = []
    page_number = 0
    threshold = 1
    common_fonts = most_common_fonts(file, page_number, threshold)
    font_ratios_dict = font_ratios(file, page_number)
    for page in file['pages']:
        for element in page['elements']:
            walk(file, page_number, element, lines, common_fonts, font_ratios_dict)
    return lines
parser = argparse.ArgumentParser(description='Extracts features to csv from .json files using .md files as labels')
parser.add_argument('md_dir', help='folder containing the .md files (labels)')
parser.add_argument('json_dir', help='folder containing the .json files (data)')
parser.add_argument('out_dir', help='folder in which to save the .csv files')
args = parser.parse_args()
paths = os.listdir(args.json_dir)
for path in paths:
    END_JSON_PATH = '.pdf.json'
    if path.endswith(END_JSON_PATH):
        print(path)
        with open(os.path.join(args.json_dir, path), mode='r', encoding='utf8') as f:
            file = json.load(f)
        with open(os.path.join(args.md_dir, path.replace(END_JSON_PATH, '.md')), mode='r', encoding='utf8') as f:
            md = f.readlines()
        contract = extract_lines(file)
        for md_line in md:
            if md_line.startswith('#'):
                level = len(md_line.split()[0])
                text_line = markdown_to_text(md_line)
                for (i, line) in enumerate(contract):
                    if contract[i][-1]:
                        continue
                    if line[0] == text_line or (line[0] in text_line and line[7] == 0 and (line[1] and line[4] or line[2] or line[3])):
                        contract[i][-3] = level
                        contract[i][-2] = 'heading'
                        contract[i][-1] = True
        if len(contract) != 0:
            contract = np.array(contract)[:, :-1]
        col_names = ['line', 'is_different_style', 'is_font_bigger', 'different_color', 'is_font_unique', 'text_case', 'word_count', 'is_number', 'nb_of_verbs', 'nb_of_nouns', 'nb_of_cardinal_numbers', 'font_size', 'is_bold', 'font_ratio', 'level', 'label']
        with open(os.path.join(args.out_dir, path.replace(END_JSON_PATH, '.csv')), newline='\n', mode='w+', encoding='utf8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(col_names)
            writer.writerows(contract)