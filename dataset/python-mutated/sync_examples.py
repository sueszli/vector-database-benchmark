import json
import os.path
import re
from collections import defaultdict
from typing import List, Tuple
example_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), 'examples'))
website_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir, 'website'))

class Example:

    def __init__(self, filename: str, title: str, description: str, tags: List[str], code: str):
        if False:
            for i in range(10):
                print('nop')
        self.name = os.path.splitext(filename)[0]
        self.slug = self.name.replace('_', '-')
        self.filename = filename
        self.title = title
        self.subtitle = description.split('\n')[0].strip()
        self.description = description
        self.tags = tags
        self.code = code

    def to_md(self):
        if False:
            return 10
        if self.tags:
            tags_meta = '\n'.join(['  - ' + x for x in self.tags])
            links = []
            for t in self.tags:
                links.append(f"<a href={{useBaseUrl('docs/examples/tags#{t}')}}>{t}</a>")
            tags_links = ' \u2002'.join(links)
            header = f"---\ntitle: {self.title}\nkeywords:\n{tags_meta}\ncustom_edit_url: null\n---\nimport useBaseUrl from '@docusaurus/useBaseUrl';\n\n        "
            footer = f'\n**Tags**: \u2002{tags_links}\n            '
        else:
            header = f'---\ntitle: {self.title}\ncustom_edit_url: null\n---\n        '
            footer = ''
        body = f"\n{self.description}\n<div className='cover' style={{{{ backgroundImage: 'url(' + require('./assets/{self.slug}.png').default + ')' }}}} />\n\n```py\n{self.code}\n```\n        "
        return header + body + footer

def read_lines(p: str) -> List[str]:
    if False:
        print('Hello World!')
    with open(p) as f:
        return f.readlines()

def read_file(p: str) -> str:
    if False:
        print('Hello World!')
    with open(p) as f:
        return f.read()

def write_file(p: str, txt: str) -> str:
    if False:
        return 10
    with open(p, 'w') as f:
        f.write(txt)
    return txt

def strip_comment(line: str) -> str:
    if False:
        while True:
            i = 10
    "Returns the content of a line without '#' and ' ' characters\n    remove leading '#', but preserve '#' that is part of a tag\n    example:\n    >>> '# #hello '.strip('#').strip()\n    '#hello'\n    "
    return line.strip('#').strip()

def parse_tags(description: str) -> Tuple[str, List[str]]:
    if False:
        while True:
            i = 10
    "Creates tags from description.\n    Accepts a description containing tags and returns a (new_description, tags) tuple.\n    The convention for tags:\n    1. Any valid twitter hashtag\n    For example, accept a description in any of the following forms\n    1. Use a checklist to group a set of related checkboxes. #form #checkbox #checklist\n    2. Use a checklist to group a set of related checkboxes.\n       #form #checkbox #checklist\n    3. Use a #checklist to group a set of related checkboxes.\n       #form #checkbox\n    and return\n    ('Use a checklist to group a set of related checkboxes.', ['checkbox', 'checklist', 'form']). The list of tags will\n    be sorted and all tags will be converted to lowercase.\n    Args:\n        description: Complete description of an example.\n    Returns:\n        A tuple of new_description and a sorted list of tags. new_description is created by removing the '#' characters\n        from the description.\n    "
    hashtag_regex_pattern = '(\\s+)#(\\w*[a-zA-Z]+\\w*)\\b'
    pattern = re.compile(hashtag_regex_pattern)
    matches = pattern.findall(' ' + description)
    tags = sorted(list(set([x[-1].lower() for x in matches])))
    new_d = pattern.sub('\\1\\2', ' ' + description)
    (*lines, last_line) = new_d.strip().splitlines()
    last_line_has_tags_only = len(last_line.strip()) > 1 and all([x.strip().lower() in tags for x in last_line.split()])
    if last_line_has_tags_only:
        return ('\n'.join(lines), tags)
    (*sentences, last_sentence) = last_line.split('. ')
    last_sentence_has_tags_only = len(last_sentence.strip()) > 1 and all([x.strip().lower() in tags for x in last_sentence.split()])
    if last_sentence_has_tags_only:
        lines.extend(sentences)
        return ('\n'.join(lines) + '.', tags)
    lines.append(last_line)
    return ('\n'.join(lines), tags)

def load_example(filename: str) -> Example:
    if False:
        return 10
    contents = read_file(os.path.join(example_dir, filename))
    parts = contents.split('---', maxsplit=1)
    (header, code) = (parts[0].strip().splitlines(), parts[1].strip())
    (title, description) = (strip_comment(header[0]), [strip_comment(x) for x in header[1:]])
    (new_description, tags) = parse_tags('\n'.join(description))
    return Example(filename, title, new_description, tags, code)

def make_toc(examples: List[Example]):
    if False:
        print('Hello World!')
    return "---\ntitle: All Examples\nslug: /examples/all\ncustom_edit_url: null\n---\nimport useBaseUrl from '@docusaurus/useBaseUrl';\n\n" + '\n\n'.join([f"- <a href={{useBaseUrl('docs/examples/{e.slug}')}}>{e.title}</a>: {e.subtitle}" for e in examples])

def make_gallery_thumbnail(e: Example):
    if False:
        return 10
    return f"<a className='thumbnail' href={{useBaseUrl('docs/examples/{e.slug}')}}><div style={{{{backgroundImage:'url(' + require('./assets/{e.slug}.png').default + ')'}}}}></div>{e.title}</a>"

def make_gallery(examples: List[Example]):
    if False:
        return 10
    return "---\ntitle: Gallery\nslug: /examples\ncustom_edit_url: null\n---\nimport useBaseUrl from '@docusaurus/useBaseUrl';\n\n" + '\n' + '\n\n'.join([make_gallery_thumbnail(e) for e in examples])

def make_tag_group(tag: str, examples: List[Example]) -> str:
    if False:
        for i in range(10):
            print('nop')
    sub_heading = f'### {tag}\n'
    example_links = []
    for e in examples:
        example_links.append(f"<a href={{useBaseUrl('docs/examples/{e.slug}')}}>{e.title}</a>")
    return sub_heading + ' \u2002'.join(example_links) + '\n'

def index_examples(examples: List[Example]):
    if False:
        while True:
            i = 10
    tags_dict = defaultdict(list)
    for e in examples:
        for t in e.tags:
            tags_dict[t].append(e)
    return sorted([(t, sorted(e, key=lambda x: x.title)) for (t, e) in tags_dict.items()])

def make_examples_by_tag(examples: List[Example]):
    if False:
        i = 10
        return i + 15
    tags = index_examples(examples)
    return "---\ntitle: Examples by Tag\nslug: /examples/tags\ncustom_edit_url: null\n---\nimport useBaseUrl from '@docusaurus/useBaseUrl';\n\n" + '\n' + '\n\n'.join([make_tag_group(t, e) for (t, e) in tags])

def read_filenames(src: str):
    if False:
        while True:
            i = 10
    return [line.strip() for line in read_lines(os.path.join(example_dir, src)) if not line.strip().startswith('#')]

def main():
    if False:
        i = 10
        return i + 15
    filenames = read_filenames('tour.conf') + read_filenames('web_only.conf')
    examples = [load_example(filename) for filename in filenames]
    example_md_dir = os.path.join(website_dir, 'docs', 'examples')
    thumbnail_dir = os.path.join(example_md_dir, 'assets')
    for f in os.listdir(example_md_dir):
        if f.endswith('.md'):
            os.remove(os.path.join(example_md_dir, f))
    for e in examples:
        md = e.to_md()
        write_file(os.path.join(example_md_dir, f'{e.slug}.md'), md)
        if not os.path.exists(os.path.join(thumbnail_dir, f'{e.slug}.png')):
            print(f'*** ALERT: no thumbnail found for example "{e.slug}"')
    example_items = [dict(slug=e.slug) for e in examples]
    example_items.insert(0, dict(slug='index'))
    example_items.insert(1, dict(slug='all'))
    example_items.insert(2, dict(slug='tags'))
    write_file(os.path.join(website_dir, 'examples.js'), f'module.exports={json.dumps(example_items)}')
    write_file(os.path.join(example_md_dir, 'index.md'), make_gallery(examples))
    write_file(os.path.join(example_md_dir, 'all.md'), make_toc(examples))
    write_file(os.path.join(example_md_dir, 'tags.md'), make_examples_by_tag(examples))
if __name__ == '__main__':
    main()