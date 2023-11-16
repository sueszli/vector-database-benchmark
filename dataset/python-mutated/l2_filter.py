import re
from tribler.core.utilities.install_dir import get_lib_path
with open(get_lib_path() / 'components' / 'metadata_store' / 'category_filter' / 'level2.regex', encoding='utf-8') as f:
    regex = f.read().strip()
    stoplist_expression = re.compile(regex, re.IGNORECASE)

def is_forbidden(txt):
    if False:
        return 10
    return bool(stoplist_expression.search(txt))