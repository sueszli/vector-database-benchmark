import autocommand
import inflect
from more_itertools import always_iterable
import jaraco.text

def report_newlines(filename):
    if False:
        i = 10
        return i + 15
    "\n    Report the newlines in the indicated file.\n\n    >>> tmp_path = getfixture('tmp_path')\n    >>> filename = tmp_path / 'out.txt'\n    >>> _ = filename.write_text('foo\\nbar\\n', newline='')\n    >>> report_newlines(filename)\n    newline is '\\n'\n    >>> filename = tmp_path / 'out.txt'\n    >>> _ = filename.write_text('foo\\nbar\\r\\n', newline='')\n    >>> report_newlines(filename)\n    newlines are ('\\n', '\\r\\n')\n    "
    newlines = jaraco.text.read_newlines(filename)
    count = len(tuple(always_iterable(newlines)))
    engine = inflect.engine()
    print(engine.plural_noun('newline', count), engine.plural_verb('is', count), repr(newlines))
autocommand.autocommand(__name__)(report_newlines)