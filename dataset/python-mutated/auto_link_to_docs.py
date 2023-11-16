"""
Process docs-links.json and updates all READMEs and replaces

<!-- auto-doc-link --><!-- end-auto-doc-link -->

With a generated list of documentation backlinks.
"""
from collections import defaultdict
import json
import os
import re
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DOC_SITE_ROOT = 'https://cloud.google.com'
AUTO_DOC_LINK_EXP = re.compile('<!-- auto-doc-link -->.*?<!-- end-auto-doc-link -->\\n', re.DOTALL)

def invert_docs_link_map(docs_links):
    if False:
        while True:
            i = 10
    '\n    The docs links map is in this format:\n\n        {\n            "doc_path": [\n                "file_path",\n            ]\n        }\n\n    This transforms it to:\n\n        {\n            "file_path": [\n                "doc_path",\n            ]\n        }\n    '
    files_to_docs = defaultdict(list)
    for (doc, files) in docs_links.iteritems():
        for file in files:
            files_to_docs[file].append(doc)
            files_to_docs[file] = list(set(files_to_docs[file]))
    return files_to_docs

def collect_docs_for_readmes(files_to_docs):
    if False:
        for i in range(10):
            print('nop')
    "\n    There's a one-to-many relationship between readmes and files. This method\n    finds the readme for each file and consolidates all docs references.\n    "
    readmes_to_docs = defaultdict(list)
    for (file, docs) in files_to_docs.iteritems():
        readme = get_readme_path(file)
        readmes_to_docs[readme].extend(docs)
        readmes_to_docs[readme] = list(set(readmes_to_docs[readme]))
    return readmes_to_docs

def linkify(docs):
    if False:
        while True:
            i = 10
    'Adds the documentation site root to doc paths, creating a full URL.'
    return [DOC_SITE_ROOT + x for x in docs]

def replace_contents(file_path, regex, new_content):
    if False:
        print('Hello World!')
    with open(file_path, 'r+') as f:
        content = f.read()
        content = regex.sub(new_content, content)
        f.seek(0)
        f.write(content)

def get_readme_path(file_path):
    if False:
        while True:
            i = 10
    'Gets the readme for an associated sample file, basically just the\n    README.md in the same directory.'
    dir = os.path.dirname(file_path)
    readme = os.path.join(REPO_ROOT, dir, 'README.md')
    return readme

def generate_doc_link_statement(docs):
    if False:
        print('Hello World!')
    links = linkify(docs)
    if len(links) == 1:
        return '<!-- auto-doc-link -->\nThese samples are used on the following documentation page:\n\n> {}\n\n<!-- end-auto-doc-link -->\n'.format(links.pop())
    else:
        return '<!-- auto-doc-link -->\nThese samples are used on the following documentation pages:\n\n>\n{}\n\n<!-- end-auto-doc-link -->\n'.format('\n'.join(['* {}'.format(x) for x in links]))

def update_readme(readme_path, docs):
    if False:
        while True:
            i = 10
    if not os.path.exists(readme_path):
        print("{} doesn't exist".format(readme_path))
        return
    replace_contents(readme_path, AUTO_DOC_LINK_EXP, generate_doc_link_statement(docs))
    print('Updated {}'.format(readme_path))

def main():
    if False:
        for i in range(10):
            print('nop')
    docs_links = json.load(open(os.path.join(REPO_ROOT, 'scripts', 'resources', 'docs-links.json'), 'r'))
    files_to_docs = invert_docs_link_map(docs_links)
    readmes_to_docs = collect_docs_for_readmes(files_to_docs)
    for (readme, docs) in readmes_to_docs.iteritems():
        update_readme(readme, docs)
if __name__ == '__main__':
    main()