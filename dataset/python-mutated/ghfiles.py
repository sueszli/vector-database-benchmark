import os.path as path
import subprocess
import shlex
from sphinx.util import logging
from docutils import nodes
logger = logging.getLogger(__name__)
top = subprocess.check_output(shlex.split('git rev-parse --show-toplevel')).strip().decode('utf-8')

def make_ref(text):
    if False:
        return 10
    ' Make hyperlink to Github '
    full_path = path.join(top, text)
    if path.isfile(full_path):
        ref = 'https://www.github.com/numba/numba/blob/main/' + text
    elif path.isdir(full_path):
        ref = 'https://www.github.com/numba/numba/tree/main/' + text
    else:
        logger.warn('Failed to find file in repomap: ' + text)
        ref = 'https://www.github.com/numba/numba'
    return ref

def intersperse(lst, item):
    if False:
        while True:
            i = 10
    ' Insert item between each item in lst.\n\n    Copied under CC-BY-SA from stackoverflow at:\n\n    https://stackoverflow.com/questions/5920643/\n    add-an-item-between-each-item-already-in-the-list\n\n    '
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result

def ghfile_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    if False:
        for i in range(10):
            print('nop')
    ' Emit hyperlink nodes for a given file in repomap. '
    my_nodes = []
    if '{' in text:
        base = text[:text.find('.') + 1]
        exts = text[text.find('{') + 1:text.find('}')].split(',')
        for e in exts:
            node = nodes.reference(rawtext, base + e, refuri=make_ref(base + e), **options)
            my_nodes.append(node)
    elif '*' in text:
        ref = path.dirname(text) + path.sep
        node = nodes.reference(rawtext, text, refuri=make_ref(ref), **options)
        my_nodes.append(node)
    else:
        node = nodes.reference(rawtext, text, refuri=make_ref(text), **options)
        my_nodes.append(node)
    if len(my_nodes) > 1:
        my_nodes = intersperse(my_nodes, nodes.Text(' | '))
    return (my_nodes, [])

def setup(app):
    if False:
        print('Hello World!')
    logger.info('Initializing ghfiles plugin')
    app.add_role('ghfile', ghfile_role)
    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata