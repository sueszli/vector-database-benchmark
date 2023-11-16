"""
Provides a formatter that generates Sphinx-based documentation
of available step definitions (step implementations).

TODO:
  * Post-processor for step docstrings.
  * Solution for requires: table, text
  * i18n keywords

.. seealso::
    http://sphinx-doc.org/

.. note:: REQUIRES docutils
    :mod:`docutils` are needed to generate step-label for step references.
"""
from __future__ import absolute_import, print_function
from operator import attrgetter
import inspect
import os.path
import sys
from behave.formatter.steps import AbstractStepsFormatter
from behave.formatter import sphinx_util
from behave.model import Table
try:
    from docutils.nodes import fully_normalize_name
    has_docutils = True
except ImportError:
    has_docutils = False

class StepsModule(object):
    """
    Value object to keep track of step definitions that belong to same module.
    """

    def __init__(self, module_name, step_definitions=None):
        if False:
            while True:
                i = 10
        self.module_name = module_name
        self.step_definitions = step_definitions or []
        self._name = None
        self._filename = None

    @property
    def name(self):
        if False:
            while True:
                i = 10
        if self._name is None:
            module = self.module
            if module:
                module_name = self.module.__name__
            else:
                module_name = self.module_name
            self._name = module_name
        return self._name

    @property
    def filename(self):
        if False:
            while True:
                i = 10
        if not self._filename:
            if self.step_definitions:
                filename = inspect.getfile(self.step_definitions[0].func)
                self._filename = os.path.relpath(filename)
        return self._filename

    @property
    def module(self):
        if False:
            while True:
                i = 10
        if self.step_definitions:
            return inspect.getmodule(self.step_definitions[0].func)
        return sys.modules.get(self.module_name)

    @property
    def module_doc(self):
        if False:
            for i in range(10):
                print('nop')
        module = self.module
        if module:
            return inspect.getdoc(module)
        return None

    def append(self, step_definition):
        if False:
            i = 10
            return i + 15
        self.step_definitions.append(step_definition)

class SphinxStepsDocumentGenerator(object):
    """
    Provides document generator class that generates Sphinx-based
    documentation for step definitions. The primary purpose is to:

      * help the step-library provider/writer
      * simplify self-documentation of step-libraries

    EXAMPLE:
        step_definitions = ...  # Collect from step_registry
        doc_generator = SphinxStepsDocumentGenerator(step_definitions, "output")
        doc_generator.write_docs()

    .. seealso:: http://sphinx-doc.org/
    """
    default_step_definition_doc = '.. todo::\n    Step definition description is missing.\n'
    shows_step_module_info = True
    shows_step_module_overview = True
    make_step_index_entries = True
    make_step_labels = has_docutils
    document_separator = '# -- DOCUMENT-END ' + '-' * 60
    step_document_prefix = 'step_module.'
    step_heading_prefix = '**Step:** '

    def __init__(self, step_definitions, destdir=None, stream=None):
        if False:
            print('Hello World!')
        self.step_definitions = step_definitions
        self.destdir = destdir
        self.stream = stream
        self.document = None

    @property
    def stdout_mode(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Indicates that output towards stdout should be used.\n        '
        return self.stream is not None

    @staticmethod
    def describe_step_definition(step_definition, step_type=None):
        if False:
            return 10
        if not step_type:
            step_type = step_definition.step_type or 'step'
        if step_type == 'step':
            step_type_text = 'Given/When/Then'
        else:
            step_type_text = step_type.capitalize()
        step_text = step_definition.pattern
        if '`' in step_text:
            step_text = step_text.replace('`', '\\`')
        return u'%s %s' % (step_type_text, step_text)

    def ensure_destdir_exists(self):
        if False:
            while True:
                i = 10
        assert self.destdir
        if os.path.isfile(self.destdir):
            print('OOPS: remove %s' % self.destdir)
            os.remove(self.destdir)
        if not os.path.exists(self.destdir):
            os.makedirs(self.destdir)

    def ensure_document_is_closed(self):
        if False:
            i = 10
            return i + 15
        if self.document and (not self.stdout_mode):
            self.document.close()
            self.document = None

    def discover_step_modules(self):
        if False:
            i = 10
            return i + 15
        step_modules_map = {}
        for step_definition in self.step_definitions:
            assert step_definition.step_type is not None
            step_filename = step_definition.location.filename
            step_module = step_modules_map.get(step_filename, None)
            if not step_module:
                filename = inspect.getfile(step_definition.func)
                module_name = inspect.getmodulename(filename)
                assert module_name, 'step_definition: %s' % step_definition.location
                step_module = StepsModule(module_name)
                step_modules_map[step_filename] = step_module
            step_module.append(step_definition)
        step_modules = sorted(step_modules_map.values(), key=attrgetter('name'))
        for module in step_modules:
            step_definitions = sorted(module.step_definitions, key=attrgetter('location'))
            module.step_definitions = step_definitions
        return step_modules

    def create_document(self, filename):
        if False:
            print('Hello World!')
        if not (filename.endswith('.rst') or filename.endswith('.txt')):
            filename += '.rst'
        if self.stdout_mode:
            stream = self.stream
            document = sphinx_util.DocumentWriter(stream, should_close=False)
        else:
            self.ensure_destdir_exists()
            filename = os.path.join(self.destdir, filename)
            document = sphinx_util.DocumentWriter.open(filename)
        return document

    def write_docs(self):
        if False:
            while True:
                i = 10
        step_modules = self.discover_step_modules()
        self.write_step_module_index(step_modules)
        for step_module in step_modules:
            self.write_step_module(step_module)
        return len(step_modules)

    def write_step_module_index(self, step_modules, filename='index.rst'):
        if False:
            for i in range(10):
                print('nop')
        document = self.create_document(filename)
        document.write('.. _docid.steps:\n\n')
        document.write_heading('Step Definitions')
        document.write('The following step definitions are provided here.\n\n----\n\n')
        entries = sorted([self.step_document_prefix + module.name for module in step_modules])
        document.write_toctree(entries, maxdepth=1)
        document.close()
        if self.stdout_mode:
            sys.stdout.write('\n%s\n' % self.document_separator)

    def write_step_module(self, step_module):
        if False:
            while True:
                i = 10
        self.ensure_document_is_closed()
        document_name = self.step_document_prefix + step_module.name
        self.document = self.create_document(document_name)
        self.document.write('.. _docid.steps.%s:\n' % step_module.name)
        self.document.write_heading(step_module.name, index_id=step_module.name)
        if self.shows_step_module_info:
            self.document.write(':Module:   %s\n' % step_module.name)
            self.document.write(':Filename: %s\n' % step_module.filename)
            self.document.write('\n')
        if step_module.module_doc:
            module_doc = step_module.module_doc.strip()
            self.document.write('%s\n\n' % module_doc)
        if self.shows_step_module_overview:
            self.document.write_heading('Step Overview', level=1)
            self.write_step_module_overview(step_module.step_definitions)
        self.document.write_heading('Step Definitions', level=1)
        for step_definition in step_module.step_definitions:
            self.write_step_definition(step_definition)
        self.document.close()
        self.document = None
        if self.stdout_mode:
            sys.stdout.write('\n%s\n' % self.document_separator)

    def write_step_module_overview(self, step_definitions):
        if False:
            return 10
        assert self.document
        headings = [u'Step Definition', u'Given', u'When', u'Then', u'Step']
        table = Table(headings)
        step_type_cols = {'given': [u'  x', u'  ', u'  ', u'  '], 'when': [u'  ', u'  x', u'  ', u'  '], 'then': [u'  ', u'  ', u'  x', u'  '], 'step': [u'  x', u'  x', u'  x', u'  x']}
        for step_definition in step_definitions:
            row = [self.describe_step_definition(step_definition)]
            row.extend(step_type_cols[step_definition.step_type])
            table.add_row(row)
        self.document.write_table(table)

    @staticmethod
    def make_step_definition_index_id(step):
        if False:
            for i in range(10):
                print('nop')
        if step.step_type == 'step':
            index_kinds = ('Given', 'When', 'Then', 'Step')
        else:
            keyword = step.step_type.capitalize()
            index_kinds = (keyword,)
        schema = 'single: %s%s; %s %s'
        index_parts = []
        for index_kind in index_kinds:
            keyword = index_kind
            word = ' step'
            if index_kind == 'Step':
                keyword = 'Given/When/Then'
                word = ''
            part = schema % (index_kind, word, keyword, step.pattern)
            index_parts.append(part)
        joiner = '\n    '
        return joiner + joiner.join(index_parts)

    def make_step_definition_doc(self, step):
        if False:
            print('Hello World!')
        doc = inspect.getdoc(step.func)
        if not doc:
            doc = self.default_step_definition_doc
        doc = doc.strip()
        return doc

    def write_step_definition(self, step):
        if False:
            for i in range(10):
                print('nop')
        assert self.document
        step_text = self.describe_step_definition(step)
        if step_text.startswith('* '):
            step_text = step_text[2:]
        index_id = None
        if self.make_step_index_entries:
            index_id = self.make_step_definition_index_id(step)
        heading = step_text
        step_label = None
        if self.step_heading_prefix:
            heading = self.step_heading_prefix + step_text
        if has_docutils and self.make_step_labels:
            step_label = fully_normalize_name(step_text)
        self.document.write_heading(heading, level=2, index_id=index_id, label=step_label)
        step_definition_doc = self.make_step_definition_doc(step)
        self.document.write('%s\n' % step_definition_doc)
        self.document.write('\n')

class SphinxStepsFormatter(AbstractStepsFormatter):
    """
    Provides formatter class that generates Sphinx-based documentation
    for all registered step definitions. The primary purpose is to:

      * help the step-library provider/writer
      * simplify self-documentation of step-libraries

    .. note::
        Supports dry-run mode.
        Supports destination directory mode to write multiple documents.
    """
    name = 'sphinx.steps'
    description = 'Generate sphinx-based documentation for step definitions.'
    doc_generator_class = SphinxStepsDocumentGenerator

    def __init__(self, stream_opener, config):
        if False:
            for i in range(10):
                print('nop')
        super(SphinxStepsFormatter, self).__init__(stream_opener, config)
        self.destdir = stream_opener.name

    @property
    def step_definitions(self):
        if False:
            i = 10
            return i + 15
        'Derive step definitions from step-registry.'
        steps = []
        for (step_type, step_definitions) in self.step_registry.steps.items():
            for step in step_definitions:
                step.step_type = step_type
                steps.append(step)
        return steps

    def close(self):
        if False:
            while True:
                i = 10
        'Called at end of test run.'
        if not self.step_registry:
            self.discover_step_definitions()
        self.report()

    def create_document_generator(self):
        if False:
            return 10
        generator_class = self.doc_generator_class
        if self.stdout_mode:
            return generator_class(self.step_definitions, stream=self.stream)
        return generator_class(self.step_definitions, destdir=self.destdir)

    def report(self):
        if False:
            i = 10
            return i + 15
        document_generator = self.create_document_generator()
        document_counts = document_generator.write_docs()
        if not self.stdout_mode:
            msg = "%s: Written %s document(s) into directory '%s'.\n"
            sys.stdout.write(msg % (self.name, document_counts, self.destdir))