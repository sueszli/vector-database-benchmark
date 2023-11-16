"""Converter from CHANGELOG.md (Markdown) to HTML suitable for a mypy blog post.

How to use:

1. Write release notes in CHANGELOG.md.
2. Make sure the heading for the next release is of form `## Mypy X.Y`.
2. Run `misc/gen_blog_post_html.py X.Y > target.html`.
4. Manually inspect and tweak the result.

Notes:

* There are some fragile assumptions. Double check the output.
"""
import argparse
import html
import os
import re
import sys

def format_lists(h: str) -> str:
    if False:
        i = 10
        return i + 15
    a = h.splitlines()
    r = []
    i = 0
    bullets = ('- ', '* ', ' * ')
    while i < len(a):
        if a[i].startswith(bullets):
            r.append('<p><ul>')
            while i < len(a) and a[i].startswith(bullets):
                r.append('<li>%s' % a[i][2:].lstrip())
                i += 1
            r.append('</ul>')
        else:
            r.append(a[i])
            i += 1
    return '\n'.join(r)

def format_code(h: str) -> str:
    if False:
        while True:
            i = 10
    a = h.splitlines()
    r = []
    i = 0
    while i < len(a):
        if a[i].startswith('    ') or a[i].startswith('```'):
            indent = a[i].startswith('    ')
            if not indent:
                i += 1
            r.append('<pre>')
            while i < len(a) and (indent and a[i].startswith('    ') or (not indent and (not a[i].startswith('```')))):
                line = a[i].replace('&gt;', '>').replace('&lt;', '<')
                if not indent:
                    line = '    ' + line
                r.append(html.escape(line))
                i += 1
            r.append('</pre>')
            if not indent and a[i].startswith('```'):
                i += 1
        else:
            r.append(a[i])
            i += 1
    return '\n'.join(r)

def convert(src: str) -> str:
    if False:
        print('Hello World!')
    h = src
    h = re.sub('<', '&lt;', h)
    h = re.sub('>', '&gt;', h)
    h = re.sub('^## (Mypy [0-9.]+)', '<h1>\\1 Released</h1>', h, flags=re.MULTILINE)
    h = re.sub('\\n#### ([A-Z`].*)\\n', '\\n<h2>\\1</h2>\\n', h)
    h = re.sub('\\n\\*\\*([A-Z_`].*)\\*\\*\\n', '\\n<h3>\\1</h3>\\n', h)
    h = re.sub('\\n`\\*\\*([A-Z_`].*)\\*\\*\\n', '\\n<h3>`\\1</h3>\\n', h)
    h = re.sub('`\\*\\*`', '<tt>**</tt>', h)
    h = re.sub('\\n([A-Z])', '\\n<p>\\1', h)
    h = format_lists(h)
    h = format_code(h)
    h = re.sub('`([^`]+)`', '<tt>\\1</tt>', h)
    h = re.sub('\\*\\*\\*\\*', '', h)
    h = re.sub('\\*\\*([A-Za-z].*?)\\*\\*', ' <b>\\1</b>', h)
    h = re.sub(' \\*([A-Za-z].*?)\\*', ' <i>\\1</i>', h)
    h = re.sub('\\[(#[0-9]+)\\]\\(https://github.com/python/mypy/pull/[0-9]+/?\\)', '\\1', h)
    h = re.sub('\\((#[0-9]+)\\) +\\(([^)]+)\\)', '(\\2, \\1)', h)
    h = re.sub('fixes #([0-9]+)', 'fixes issue <a href="https://github.com/python/mypy/issues/\\1">\\1</a>', h)
    h = re.sub('#([0-9]+)', 'PR <a href="https://github.com/python/mypy/pull/\\1">\\1</a>', h)
    h = re.sub('\\) \\(PR', ', PR', h)
    h = re.sub('\\[([^]]*)\\]\\(([^)]*)\\)', '<a href="\\2">\\1</a>', h)
    h = re.sub('contributors to typeshed:', 'contributors to <a href="https://github.com/python/typeshed">typeshed</a>:', h)
    h = '<html>\n<meta charset="utf-8" />\n<body>\n' + h + '</body>\n</html>'
    return h

def extract_version(src: str, version: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    a = src.splitlines()
    i = 0
    heading = f'## Mypy {version}'
    while i < len(a):
        if a[i].strip() == heading:
            break
        i += 1
    else:
        raise RuntimeError(f"Can't find heading {heading!r}")
    j = i + 1
    while not a[j].startswith('## '):
        j += 1
    return '\n'.join(a[i:j])

def main() -> None:
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Generate HTML release blog post based on CHANGELOG.md and write to stdout.')
    parser.add_argument('version', help='mypy version, in form X.Y or X.Y.Z')
    args = parser.parse_args()
    version: str = args.version
    if not re.match('[0-9]+(\\.[0-9]+)+$', version):
        sys.exit(f'error: Version must be of form X.Y or X.Y.Z, not {version!r}')
    changelog_path = os.path.join(os.path.dirname(__file__), os.path.pardir, 'CHANGELOG.md')
    src = open(changelog_path).read()
    src = extract_version(src, version)
    dst = convert(src)
    sys.stdout.write(dst)
if __name__ == '__main__':
    main()