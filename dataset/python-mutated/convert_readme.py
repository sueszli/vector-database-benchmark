"""Remove raw HTML from README.rst to make it compatible with PyPI on
dist upload.
"""
import argparse
import re
summary = "Quick links\n===========\n\n- `Home page <https://github.com/giampaolo/psutil>`_\n- `Install <https://github.com/giampaolo/psutil/blob/master/INSTALL.rst>`_\n- `Documentation <http://psutil.readthedocs.io>`_\n- `Download <https://pypi.org/project/psutil/#files>`_\n- `Forum <http://groups.google.com/group/psutil/topics>`_\n- `StackOverflow <https://stackoverflow.com/questions/tagged/psutil>`_\n- `Blog <https://gmpy.dev/tags/psutil>`_\n- `What's new <https://github.com/giampaolo/psutil/blob/master/HISTORY.rst>`_\n"
funding = 'Sponsors\n========\n\n.. image:: https://github.com/giampaolo/psutil/raw/master/docs/_static/tidelift-logo.png\n  :width: 200\n  :alt: Alternative text\n\n`Add your logo <https://github.com/sponsors/giampaolo>`__.\n\nExample usages'

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=str)
    args = parser.parse_args()
    with open(args.file) as f:
        data = f.read()
    data = re.sub('.. raw:: html\\n+\\s+<div align[\\s\\S]*?/div>', summary, data)
    data = re.sub('Sponsors\\n========[\\s\\S]*?Example usages', funding, data)
    print(data)
if __name__ == '__main__':
    main()