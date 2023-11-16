"""Converts Jupyter Notebooks to Jekyll compliant blog posts"""
from datetime import datetime
import re, os, logging
from nbdev import export2html
from nbdev.export2html import Config, Path, _to_html, _re_block_notes
from fast_template import rename_for_jekyll
warnings = set()

def _nb2htmlfname(nb_path, dest=None):
    if False:
        print('Hello World!')
    fname = rename_for_jekyll(nb_path, warnings=warnings)
    if dest is None:
        dest = Config().doc_path
    return Path(dest) / fname
for (original, new) in warnings:
    print(f'{original} has been renamed to {new} to be complaint with Jekyll naming conventions.\n')
export2html._nb2htmlfname = _nb2htmlfname
export2html.notebook2html(fname='_notebooks/*.ipynb', dest='_posts/', template_file='/fastpages/fastpages.tpl', execute=False)