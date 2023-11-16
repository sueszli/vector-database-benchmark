from datetime import datetime
import re, os
from pathlib import Path
from typing import Tuple, Set
_re_blog_date = re.compile('([12]\\d{3}-(0[1-9]|1[0-2])-(0[1-9]|[12]\\d|3[01])-)')
_re_numdash = re.compile('(^[-\\d]+)')

def rename_for_jekyll(nb_path: Path, warnings: Set[Tuple[str, str]]=None) -> str:
    if False:
        while True:
            i = 10
    "\n    Return a Path's filename string appended with its modified time in YYYY-MM-DD format.\n    "
    assert nb_path.exists(), f'{nb_path} could not be found.'
    if _re_blog_date.match(nb_path.name):
        return nb_path.with_suffix('.md').name.replace(' ', '-')
    else:
        clean_name = _re_numdash.sub('', nb_path.with_suffix('.md').name).replace(' ', '-')
        mdate = os.path.getmtime(nb_path) - 86400
        dtnm = datetime.fromtimestamp(mdate).strftime('%Y-%m-%d-') + clean_name
        assert _re_blog_date.match(dtnm), f'{dtnm} is not a valid name, filename must be pre-pended with YYYY-MM-DD-'
        if warnings:
            warnings.add((nb_path, dtnm))
        return dtnm