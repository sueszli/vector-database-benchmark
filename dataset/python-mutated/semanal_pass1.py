"""Block/import reachability analysis."""
from __future__ import annotations
from mypy.nodes import AssertStmt, AssignmentStmt, Block, ClassDef, ExpressionStmt, ForStmt, FuncDef, IfStmt, Import, ImportAll, ImportFrom, MatchStmt, MypyFile, ReturnStmt
from mypy.options import Options
from mypy.reachability import assert_will_always_fail, infer_reachability_of_if_statement, infer_reachability_of_match_statement
from mypy.traverser import TraverserVisitor

class SemanticAnalyzerPreAnalysis(TraverserVisitor):
    """Analyze reachability of blocks and imports and other local things.

    This runs before semantic analysis, so names have not been bound. Imports are
    also not resolved yet, so we can only access the current module.

    This determines static reachability of blocks and imports due to version and
    platform checks, among others.

    The main entry point is 'visit_file'.

    Reachability of imports needs to be determined very early in the build since
    this affects which modules will ultimately be processed.

    Consider this example:

      import sys

      def do_stuff() -> None:
          if sys.version_info >= (3, 10):
              import xyz  # Only available in Python 3.10+
              xyz.whatever()
          ...

    The block containing 'import xyz' is unreachable in Python 3 mode. The import
    shouldn't be processed in Python 3 mode, even if the module happens to exist.
    """

    def visit_file(self, file: MypyFile, fnam: str, mod_id: str, options: Options) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.platform = options.platform
        self.cur_mod_id = mod_id
        self.cur_mod_node = file
        self.options = options
        self.is_global_scope = True
        self.skipped_lines: set[int] = set()
        for (i, defn) in enumerate(file.defs):
            defn.accept(self)
            if isinstance(defn, AssertStmt) and assert_will_always_fail(defn, options):
                if i < len(file.defs) - 1:
                    (next_def, last) = (file.defs[i + 1], file.defs[-1])
                    if last.end_line is not None:
                        self.skipped_lines |= set(range(next_def.line, last.end_line + 1))
                del file.defs[i + 1:]
                break
        file.skipped_lines = self.skipped_lines

    def visit_func_def(self, node: FuncDef) -> None:
        if False:
            for i in range(10):
                print('nop')
        old_global_scope = self.is_global_scope
        self.is_global_scope = False
        super().visit_func_def(node)
        self.is_global_scope = old_global_scope
        file_node = self.cur_mod_node
        if self.is_global_scope and file_node.is_stub and (node.name == '__getattr__') and file_node.is_package_init_file():
            file_node.is_partial_stub_package = True

    def visit_class_def(self, node: ClassDef) -> None:
        if False:
            return 10
        old_global_scope = self.is_global_scope
        self.is_global_scope = False
        super().visit_class_def(node)
        self.is_global_scope = old_global_scope

    def visit_import_from(self, node: ImportFrom) -> None:
        if False:
            while True:
                i = 10
        node.is_top_level = self.is_global_scope
        super().visit_import_from(node)

    def visit_import_all(self, node: ImportAll) -> None:
        if False:
            i = 10
            return i + 15
        node.is_top_level = self.is_global_scope
        super().visit_import_all(node)

    def visit_import(self, node: Import) -> None:
        if False:
            for i in range(10):
                print('nop')
        node.is_top_level = self.is_global_scope
        super().visit_import(node)

    def visit_if_stmt(self, s: IfStmt) -> None:
        if False:
            while True:
                i = 10
        infer_reachability_of_if_statement(s, self.options)
        for expr in s.expr:
            expr.accept(self)
        for node in s.body:
            node.accept(self)
        if s.else_body:
            s.else_body.accept(self)

    def visit_block(self, b: Block) -> None:
        if False:
            return 10
        if b.is_unreachable:
            if b.end_line is not None:
                self.skipped_lines |= set(range(b.line, b.end_line + 1))
            return
        super().visit_block(b)

    def visit_match_stmt(self, s: MatchStmt) -> None:
        if False:
            return 10
        infer_reachability_of_match_statement(s, self.options)
        for guard in s.guards:
            if guard is not None:
                guard.accept(self)
        for body in s.bodies:
            body.accept(self)

    def visit_assignment_stmt(self, s: AssignmentStmt) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def visit_expression_stmt(self, s: ExpressionStmt) -> None:
        if False:
            while True:
                i = 10
        pass

    def visit_return_stmt(self, s: ReturnStmt) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def visit_for_stmt(self, s: ForStmt) -> None:
        if False:
            while True:
                i = 10
        s.body.accept(self)
        if s.else_body is not None:
            s.else_body.accept(self)