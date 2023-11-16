"""
Try to detect suspicious constructs, resembling markup
that has leaked into the final output.

Suspicious lines are reported in a comma-separated-file,
``suspicious.csv``, located in the output directory.

The file is utf-8 encoded, and each line contains four fields:

 * document name (normalized)
 * line number in the source document
 * problematic text
 * complete line showing the problematic text in context

It is common to find many false positives. To avoid reporting them
again and again, they may be added to the ``ignored.csv`` file
(located in the configuration directory). The file has the same
format as ``suspicious.csv`` with a few differences:

  - each line defines a rule; if the rule matches, the issue
    is ignored.
  - line number may be empty (that is, nothing between the
    commas: ",,"). In this case, line numbers are ignored (the
    rule matches anywhere in the file).
  - the last field does not have to be a complete line; some
    surrounding text (never more than a line) is enough for
    context.

Rules are processed sequentially. A rule matches when:

 * document names are the same
 * problematic texts are the same
 * line numbers are close to each other (5 lines up or down)
 * the rule text is completely contained into the source line

The simplest way to create the ignored.csv file is by copying
undesired entries from suspicious.csv (possibly trimming the last
field.)

Copyright 2009 Gabriel A. Genellina

"""
import os
import re
import csv
import sys
from docutils import nodes
from sphinx.builders import Builder
import sphinx.util
detect_all = re.compile('\n    ::(?=[^=])|            # two :: (but NOT ::=)\n    :[a-zA-Z][a-zA-Z0-9]+| # :foo\n    `|                     # ` (seldom used by itself)\n    (?<!\\.)\\.\\.[ \\t]*\\w+:  # .. foo: (but NOT ... else:)\n    ', re.UNICODE | re.VERBOSE).finditer
py3 = sys.version_info >= (3, 0)

class Rule:

    def __init__(self, docname, lineno, issue, line):
        if False:
            return 10
        'A rule for ignoring issues'
        self.docname = docname
        self.lineno = lineno
        self.issue = issue
        self.line = line
        self.used = False

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '{0.docname},,{0.issue},{0.line}'.format(self)

class dialect(csv.excel):
    """Our dialect: uses only linefeed as newline."""
    lineterminator = '\n'

class CheckSuspiciousMarkupBuilder(Builder):
    """
    Checks for possibly invalid markup that may leak into the output.
    """
    name = 'suspicious'
    logger = sphinx.util.logging.getLogger('CheckSuspiciousMarkupBuilder')

    def init(self):
        if False:
            while True:
                i = 10
        self.log_file_name = os.path.join(self.outdir, 'suspicious.csv')
        open(self.log_file_name, 'w').close()
        self.load_rules(os.path.join(os.path.dirname(__file__), '..', 'susp-ignored.csv'))

    def get_outdated_docs(self):
        if False:
            print('Hello World!')
        return self.env.found_docs

    def get_target_uri(self, docname, typ=None):
        if False:
            print('Hello World!')
        return ''

    def prepare_writing(self, docnames):
        if False:
            for i in range(10):
                print('nop')
        pass

    def write_doc(self, docname, doctree):
        if False:
            for i in range(10):
                print('nop')
        self.any_issue = False
        self.docname = docname
        visitor = SuspiciousVisitor(doctree, self)
        doctree.walk(visitor)

    def finish(self):
        if False:
            return 10
        unused_rules = [rule for rule in self.rules if not rule.used]
        if unused_rules:
            self.logger.warning('Found %s/%s unused rules: %s' % (len(unused_rules), len(self.rules), ''.join((repr(rule) for rule in unused_rules))))
        return

    def check_issue(self, line, lineno, issue):
        if False:
            i = 10
            return i + 15
        if not self.is_ignored(line, lineno, issue):
            self.report_issue(line, lineno, issue)

    def is_ignored(self, line, lineno, issue):
        if False:
            i = 10
            return i + 15
        'Determine whether this issue should be ignored.'
        docname = self.docname
        for rule in self.rules:
            if rule.docname != docname:
                continue
            if rule.issue != issue:
                continue
            if rule.line not in line:
                continue
            if rule.lineno is not None and abs(rule.lineno - lineno) > 5:
                continue
            rule.used = True
            return True
        return False

    def report_issue(self, text, lineno, issue):
        if False:
            print('Hello World!')
        self.any_issue = True
        self.write_log_entry(lineno, issue, text)
        if py3:
            self.logger.warning('[%s:%d] "%s" found in "%-.120s"' % (self.docname, lineno, issue, text))
        else:
            self.logger.warning('[%s:%d] "%s" found in "%-.120s"' % (self.docname.encode(sys.getdefaultencoding(), 'replace'), lineno, issue.encode(sys.getdefaultencoding(), 'replace'), text.strip().encode(sys.getdefaultencoding(), 'replace')))
        self.app.statuscode = 1

    def write_log_entry(self, lineno, issue, text):
        if False:
            for i in range(10):
                print('nop')
        if py3:
            f = open(self.log_file_name, 'a')
            writer = csv.writer(f, dialect)
            writer.writerow([self.docname, lineno, issue, text.strip()])
            f.close()
        else:
            f = open(self.log_file_name, 'ab')
            writer = csv.writer(f, dialect)
            writer.writerow([self.docname.encode('utf-8'), lineno, issue.encode('utf-8'), text.strip().encode('utf-8')])
            f.close()

    def load_rules(self, filename):
        if False:
            print('Hello World!')
        'Load database of previously ignored issues.\n\n        A csv file, with exactly the same format as suspicious.csv\n        Fields: document name (normalized), line number, issue, surrounding text\n        '
        self.logger.info('loading ignore rules... ', nonl=1)
        self.rules = rules = []
        try:
            if py3:
                f = open(filename, 'r')
            else:
                f = open(filename, 'rb')
        except IOError:
            return
        for (i, row) in enumerate(csv.reader(f)):
            if len(row) != 4:
                raise ValueError('wrong format in %s, line %d: %s' % (filename, i + 1, row))
            (docname, lineno, issue, text) = row
            if lineno:
                lineno = int(lineno)
            else:
                lineno = None
            if not py3:
                docname = docname.decode('utf-8')
                issue = issue.decode('utf-8')
                text = text.decode('utf-8')
            rule = Rule(docname, lineno, issue, text)
            rules.append(rule)
        f.close()
        self.logger.info('done, %d rules loaded' % len(self.rules))

def get_lineno(node):
    if False:
        i = 10
        return i + 15
    'Obtain line number information for a node.'
    lineno = None
    while lineno is None and node:
        node = node.parent
        lineno = node.line
    return lineno

def extract_line(text, index):
    if False:
        i = 10
        return i + 15
    'text may be a multiline string; extract\n    only the line containing the given character index.\n\n    >>> extract_line("abc\ndefgh\ni", 6)\n    >>> \'defgh\'\n    >>> for i in (0, 2, 3, 4, 10):\n    ...   print extract_line("abc\ndefgh\ni", i)\n    abc\n    abc\n    abc\n    defgh\n    defgh\n    i\n    '
    p = text.rfind('\n', 0, index) + 1
    q = text.find('\n', index)
    if q < 0:
        q = len(text)
    return text[p:q]

class SuspiciousVisitor(nodes.GenericNodeVisitor):
    lastlineno = 0

    def __init__(self, document, builder):
        if False:
            while True:
                i = 10
        nodes.GenericNodeVisitor.__init__(self, document)
        self.builder = builder

    def default_visit(self, node):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(node, (nodes.Text, nodes.image)):
            text = node.astext()
            self.lastlineno = lineno = max(get_lineno(node) or 0, self.lastlineno)
            seen = set()
            for match in detect_all(text):
                issue = match.group()
                line = extract_line(text, match.start())
                if (issue, line) not in seen:
                    self.builder.check_issue(line, lineno, issue)
                    seen.add((issue, line))
    unknown_visit = default_visit

    def visit_document(self, node):
        if False:
            while True:
                i = 10
        self.lastlineno = 0

    def visit_comment(self, node):
        if False:
            return 10
        raise nodes.SkipNode