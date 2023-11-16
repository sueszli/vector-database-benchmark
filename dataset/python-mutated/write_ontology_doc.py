from pathlib import Path
from snips_nlu_parsers.builtin_entities import get_complete_entity_ontology, get_all_gazetteer_entities
ONTOLOGY = get_complete_entity_ontology()
ALL_GAZETTEER_ENTITIES = get_all_gazetteer_entities()
LANGUAGE_DOC_PATH = Path(__file__).parent / 'source' / 'languages.rst'
ENTITIES_DOC_PATH = Path(__file__).parent / 'source' / 'builtin_entities.rst'
GRAMMAR_ENTITY = 'Grammar Entity'
GAZETTEER_ENTITY = 'Gazetteer Entity'
LANGUAGES_DOC_HEADER = '.. _languages:\n\nSupported languages\n===================\n\nSnips NLU supports various languages. The language is specified in the\n:ref:`dataset <json_format>` in the ``"language"`` attribute.\n\nHere is the list of the supported languages along with their isocode:\n'
LANGUAGES_DOC_FOOTER = '\n\nSupport for additional languages will come in the future, stay tuned :)\n'
ENTITIES_DOC_HEADER = '.. _builtin_entities:\n\nSupported builtin entities\n==========================\n\n:ref:`Builtin entities <builtin_entity_resolution>` are entities that have\na built-in support in Snips NLU. These entities are associated to specific\nbuiltin entity parsers which provide an extra resolution step. Typically,\ndates written in natural language (``"in three days"``) are resolved into ISO\nformatted dates (``"2019-08-12 00:00:00 +02:00"``).\n\nHere is the list of supported builtin entities:\n\n'
ENTITIES_DOC_MIDDLE = "\n\nThe entity identifier (second column above) is what is used in the dataset to\nreference a builtin entity.\n\nGrammar Entity\n--------------\n\nGrammar entities, in the context of Snips NLU, correspond to entities which \ncontain significant `compositionality`_. The semantic meaning of such an \nentity is determined by the meanings of its constituent expressions and the \nrules used to combine them. Modern semantic parsers for these entities are \noften based on defining a formal grammar. In the case of Snips NLU, the parser \nused to handle these entities is `Rustling`_, a Rust adaptation of Facebook's \n`duckling`_.\n\nGazetteer Entity\n----------------\n\nGazetteer entities correspond to all the builtin entities which do not contain \nany semantic structure, as opposed to the grammar entities. For such \nentities, a `gazetteer entity parser`_ is used to perform the parsing.\n\nResults Examples\n----------------\n\nThe following sections provide examples for each builtin entity. \n\n"
ENTITIES_DOC_FOOTER = '\n.. _compositionality: https://en.wikipedia.org/wiki/Principle_of_compositionality\n.. _Rustling: https://github.com/snipsco/rustling-ontology\n.. _duckling: https://github.com/facebook/duckling\n.. _gazetteer entity parser: https://github.com/snipsco/gazetteer-entity-parser\n'
LANGUAGES_TABLE_CELL_LENGTH = 12
ENTITIES_TABLE_CELL_LENGTH = 50

def write_supported_languages(path):
    if False:
        for i in range(10):
            print('nop')
    languages = sorted([lang_ontology['language'] for lang_ontology in ONTOLOGY])
    table = _build_supported_languages_table(languages)
    content = LANGUAGES_DOC_HEADER + table + LANGUAGES_DOC_FOOTER
    with path.open(mode='w') as f:
        f.write(content)

def write_supported_builtin_entities(path):
    if False:
        print('Hello World!')
    table = _build_supported_entities_table(ONTOLOGY)
    results_examples = _build_results_examples(ONTOLOGY)
    content = ENTITIES_DOC_HEADER + table + ENTITIES_DOC_MIDDLE + results_examples + ENTITIES_DOC_FOOTER
    with path.open(mode='w') as f:
        f.write(content)

def _build_supported_languages_table(languages):
    if False:
        for i in range(10):
            print('nop')
    table = _build_table_cells(['ISO code'], LANGUAGES_TABLE_CELL_LENGTH, '=', '-')
    for language in languages:
        table += _build_table_cells([language], LANGUAGES_TABLE_CELL_LENGTH, '-')
    return table

def _build_supported_entities_table(ontology):
    if False:
        while True:
            i = 10
    en_ontology = None
    for lang_ontology in ontology:
        if lang_ontology['language'] == 'en':
            en_ontology = lang_ontology
            break
    table = _build_table_cells(['Entity', 'Identifier', 'Category', 'Supported Languages'], ENTITIES_TABLE_CELL_LENGTH, '=', '-')
    for entity in en_ontology['entities']:
        table += _build_table_cells(['`%s`_' % entity['name'], entity['label'], '`%s`_' % _category(entity['label']), ', '.join(entity['supportedLanguages'])], ENTITIES_TABLE_CELL_LENGTH, '-')
    return table

def _build_results_examples(ontology):
    if False:
        print('Hello World!')
    content = ''
    en_ontology = None
    for lang_ontology in ontology:
        if lang_ontology['language'] == 'en':
            en_ontology = lang_ontology
            break
    for entity in en_ontology['entities']:
        name = entity['name']
        title = '\n'.join([len(name) * '-', name, len(name) * '-'])
        input_examples = '\nInput examples:\n\n.. code-block:: json\n\n   [\n     %s\n   ]\n' % ',\n     '.join(['"%s"' % ex for ex in entity['examples']])
        output_examples = '\nOutput examples:\n\n.. code-block:: json\n\n   %s\n\n' % entity['resultDescription'].replace('\n', '\n   ')
        content += '\n'.join([title, input_examples, output_examples])
    return content

def _build_table_cells(contents, cell_length, bottom_sep_char, top_sep_char=None):
    if False:
        while True:
            i = 10
    cells = []
    for (i, content) in enumerate(contents):
        right_bar = ''
        right_plus = ''
        if i == len(contents) - 1:
            right_bar = '|'
            right_plus = '+'
        blank_suffix_length = cell_length - len(content) - 1
        blank_suffix = blank_suffix_length * ' '
        cell_prefix = ''
        if top_sep_char is not None:
            top_line_sep = cell_length * top_sep_char
            cell_prefix = '+%s%s\n' % (top_line_sep, right_plus)
        bottom_line_sep = cell_length * bottom_sep_char
        cell = '\n%s| %s%s%s\n+%s%s' % (cell_prefix, content, blank_suffix, right_bar, bottom_line_sep, right_plus)
        cells.append(cell)
    cell_lines = zip(*(c.split('\n') for c in cells))
    cell_lines = [''.join(line) for line in cell_lines]
    cell = '\n'.join(cell_lines)
    return cell

def _category(entity_identifier):
    if False:
        while True:
            i = 10
    if entity_identifier in ALL_GAZETTEER_ENTITIES:
        return GAZETTEER_ENTITY
    return GRAMMAR_ENTITY
if __name__ == '__main__':
    write_supported_languages(LANGUAGE_DOC_PATH)
    write_supported_builtin_entities(ENTITIES_DOC_PATH)