""" Build index from directory listing
From: https://stackoverflow.com/questions/39048654/how-to-enable-directory-indexing-on-github-pages

make_index.py </path/to/directory>
"""
INDEX_TEMPLATE = '\n<html>\n<title>Links for lief</title>\n<body>\n<h1>Links for lief</h1>\n% for name in names:\n    <a href="${base_url}/${base}/${name}">${name}</a><br />\n% endfor\n</body>\n</html>\n'
EXCLUDED = ['index.html', '.gitkeep']
BASE_URL = 'https://lief-project.github.io'
import os
import argparse
from mako.template import Template

def main():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--base')
    parser.add_argument('--output')
    args = parser.parse_args()
    fnames = [fname for fname in sorted(os.listdir(args.directory)) if fname not in EXCLUDED]
    html = Template(INDEX_TEMPLATE).render(names=fnames, base_url=BASE_URL, base=args.base)
    with open(args.output, 'w') as f:
        f.write(html)
if __name__ == '__main__':
    main()