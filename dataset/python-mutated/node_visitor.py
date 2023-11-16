import ast
import logging
import operator
from bandit.core import constants
from bandit.core import tester as b_tester
from bandit.core import utils as b_utils
LOG = logging.getLogger(__name__)

class BanditNodeVisitor:

    def __init__(self, fname, fdata, metaast, testset, debug, nosec_lines, metrics):
        if False:
            i = 10
            return i + 15
        self.debug = debug
        self.nosec_lines = nosec_lines
        self.seen = 0
        self.scores = {'SEVERITY': [0] * len(constants.RANKING), 'CONFIDENCE': [0] * len(constants.RANKING)}
        self.depth = 0
        self.fname = fname
        self.fdata = fdata
        self.metaast = metaast
        self.testset = testset
        self.imports = set()
        self.import_aliases = {}
        self.tester = b_tester.BanditTester(self.testset, self.debug, nosec_lines, metrics)
        try:
            self.namespace = b_utils.get_module_qualname_from_path(fname)
        except b_utils.InvalidModulePath:
            LOG.warning('Unable to find qualified name for module: %s', self.fname)
            self.namespace = ''
        LOG.debug('Module qualified name: %s', self.namespace)
        self.metrics = metrics

    def visit_ClassDef(self, node):
        if False:
            print('Hello World!')
        'Visitor for AST ClassDef node\n\n        Add class name to current namespace for all descendants.\n        :param node: Node being inspected\n        :return: -\n        '
        self.namespace = b_utils.namespace_path_join(self.namespace, node.name)

    def visit_FunctionDef(self, node):
        if False:
            while True:
                i = 10
        'Visitor for AST FunctionDef nodes\n\n        add relevant information about the node to\n        the context for use in tests which inspect function definitions.\n        Add the function name to the current namespace for all descendants.\n        :param node: The node that is being inspected\n        :return: -\n        '
        self.context['function'] = node
        qualname = self.namespace + '.' + b_utils.get_func_name(node)
        name = qualname.split('.')[-1]
        self.context['qualname'] = qualname
        self.context['name'] = name
        self.namespace = b_utils.namespace_path_join(self.namespace, name)
        self.update_scores(self.tester.run_tests(self.context, 'FunctionDef'))

    def visit_Call(self, node):
        if False:
            return 10
        'Visitor for AST Call nodes\n\n        add relevant information about the node to\n        the context for use in tests which inspect function calls.\n        :param node: The node that is being inspected\n        :return: -\n        '
        self.context['call'] = node
        qualname = b_utils.get_call_name(node, self.import_aliases)
        name = qualname.split('.')[-1]
        self.context['qualname'] = qualname
        self.context['name'] = name
        self.update_scores(self.tester.run_tests(self.context, 'Call'))

    def visit_Import(self, node):
        if False:
            for i in range(10):
                print('nop')
        'Visitor for AST Import nodes\n\n        add relevant information about node to\n        the context for use in tests which inspect imports.\n        :param node: The node that is being inspected\n        :return: -\n        '
        for nodename in node.names:
            if nodename.asname:
                self.import_aliases[nodename.asname] = nodename.name
            self.imports.add(nodename.name)
            self.context['module'] = nodename.name
        self.update_scores(self.tester.run_tests(self.context, 'Import'))

    def visit_ImportFrom(self, node):
        if False:
            return 10
        'Visitor for AST ImportFrom nodes\n\n        add relevant information about node to\n        the context for use in tests which inspect imports.\n        :param node: The node that is being inspected\n        :return: -\n        '
        module = node.module
        if module is None:
            return self.visit_Import(node)
        for nodename in node.names:
            if nodename.asname:
                self.import_aliases[nodename.asname] = module + '.' + nodename.name
            else:
                self.import_aliases[nodename.name] = module + '.' + nodename.name
            self.imports.add(module + '.' + nodename.name)
            self.context['module'] = module
            self.context['name'] = nodename.name
        self.update_scores(self.tester.run_tests(self.context, 'ImportFrom'))

    def visit_Constant(self, node):
        if False:
            print('Hello World!')
        'Visitor for AST Constant nodes\n\n        call the appropriate method for the node type.\n        this maintains compatibility with <3.6 and 3.8+\n\n        This code is heavily influenced by Anthony Sottile (@asottile) here:\n        https://bugs.python.org/msg342486\n\n        :param node: The node that is being inspected\n        :return: -\n        '
        if isinstance(node.value, str):
            self.visit_Str(node)
        elif isinstance(node.value, bytes):
            self.visit_Bytes(node)

    def visit_Str(self, node):
        if False:
            while True:
                i = 10
        'Visitor for AST String nodes\n\n        add relevant information about node to\n        the context for use in tests which inspect strings.\n        :param node: The node that is being inspected\n        :return: -\n        '
        self.context['str'] = node.s
        if not isinstance(node._bandit_parent, ast.Expr):
            self.context['linerange'] = b_utils.linerange(node._bandit_parent)
            self.update_scores(self.tester.run_tests(self.context, 'Str'))

    def visit_Bytes(self, node):
        if False:
            i = 10
            return i + 15
        'Visitor for AST Bytes nodes\n\n        add relevant information about node to\n        the context for use in tests which inspect strings.\n        :param node: The node that is being inspected\n        :return: -\n        '
        self.context['bytes'] = node.s
        if not isinstance(node._bandit_parent, ast.Expr):
            self.context['linerange'] = b_utils.linerange(node._bandit_parent)
            self.update_scores(self.tester.run_tests(self.context, 'Bytes'))

    def pre_visit(self, node):
        if False:
            return 10
        self.context = {}
        self.context['imports'] = self.imports
        self.context['import_aliases'] = self.import_aliases
        if self.debug:
            LOG.debug(ast.dump(node))
            self.metaast.add_node(node, '', self.depth)
        if hasattr(node, 'lineno'):
            self.context['lineno'] = node.lineno
        if hasattr(node, 'col_offset'):
            self.context['col_offset'] = node.col_offset
        if hasattr(node, 'end_col_offset'):
            self.context['end_col_offset'] = node.end_col_offset
        self.context['node'] = node
        self.context['linerange'] = b_utils.linerange(node)
        self.context['filename'] = self.fname
        self.context['file_data'] = self.fdata
        self.seen += 1
        LOG.debug('entering: %s %s [%s]', hex(id(node)), type(node), self.depth)
        self.depth += 1
        LOG.debug(self.context)
        return True

    def visit(self, node):
        if False:
            i = 10
            return i + 15
        name = node.__class__.__name__
        method = 'visit_' + name
        visitor = getattr(self, method, None)
        if visitor is not None:
            if self.debug:
                LOG.debug('%s called (%s)', method, ast.dump(node))
            visitor(node)
        else:
            self.update_scores(self.tester.run_tests(self.context, name))

    def post_visit(self, node):
        if False:
            i = 10
            return i + 15
        self.depth -= 1
        LOG.debug('%s\texiting : %s', self.depth, hex(id(node)))
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            self.namespace = b_utils.namespace_path_split(self.namespace)[0]

    def generic_visit(self, node):
        if False:
            return 10
        'Drive the visitor.'
        for (_, value) in ast.iter_fields(node):
            if isinstance(value, list):
                max_idx = len(value) - 1
                for (idx, item) in enumerate(value):
                    if isinstance(item, ast.AST):
                        if idx < max_idx:
                            item._bandit_sibling = value[idx + 1]
                        else:
                            item._bandit_sibling = None
                        item._bandit_parent = node
                        if self.pre_visit(item):
                            self.visit(item)
                            self.generic_visit(item)
                            self.post_visit(item)
            elif isinstance(value, ast.AST):
                value._bandit_sibling = None
                value._bandit_parent = node
                if self.pre_visit(value):
                    self.visit(value)
                    self.generic_visit(value)
                    self.post_visit(value)

    def update_scores(self, scores):
        if False:
            i = 10
            return i + 15
        'Score updater\n\n        Since we moved from a single score value to a map of scores per\n        severity, this is needed to update the stored list.\n        :param score: The score list to update our scores with\n        '
        for score_type in self.scores:
            self.scores[score_type] = list(map(operator.add, self.scores[score_type], scores[score_type]))

    def process(self, data):
        if False:
            i = 10
            return i + 15
        'Main process loop\n\n        Build and process the AST\n        :param lines: lines code to process\n        :return score: the aggregated score for the current file\n        '
        f_ast = ast.parse(data)
        self.generic_visit(f_ast)
        return self.scores