import shutil
from pathlib import Path
import ast_comments
from freqtrade.constants import Config

class StrategyUpdater:
    name_mapping = {'ticker_interval': 'timeframe', 'buy': 'enter_long', 'sell': 'exit_long', 'buy_tag': 'enter_tag', 'sell_reason': 'exit_reason', 'sell_signal': 'exit_signal', 'custom_sell': 'custom_exit', 'force_sell': 'force_exit', 'emergency_sell': 'emergency_exit', 'use_sell_signal': 'use_exit_signal', 'sell_profit_only': 'exit_profit_only', 'sell_profit_offset': 'exit_profit_offset', 'ignore_roi_if_buy_signal': 'ignore_roi_if_entry_signal', 'forcebuy_enable': 'force_entry_enable'}
    function_mapping = {'populate_buy_trend': 'populate_entry_trend', 'populate_sell_trend': 'populate_exit_trend', 'custom_sell': 'custom_exit', 'check_buy_timeout': 'check_entry_timeout', 'check_sell_timeout': 'check_exit_timeout'}
    otif_ot_unfilledtimeout = {'buy': 'entry', 'sell': 'exit'}
    rename_dict = {'buy': 'enter_long', 'sell': 'exit_long', 'buy_tag': 'enter_tag'}

    def start(self, config: Config, strategy_obj: dict) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Run strategy updater\n        It updates a strategy to v3 with the help of the ast-module\n        :return: None\n        '
        source_file = strategy_obj['location']
        strategies_backup_folder = Path.joinpath(config['user_data_dir'], 'strategies_orig_updater')
        target_file = Path.joinpath(strategies_backup_folder, strategy_obj['location_rel'])
        with Path(source_file).open('r') as f:
            old_code = f.read()
        if not strategies_backup_folder.is_dir():
            Path(strategies_backup_folder).mkdir(parents=True, exist_ok=True)
        shutil.copy(source_file, target_file)
        new_code = self.update_code(old_code)
        with Path(source_file).open('w') as f:
            f.write(new_code)

    def update_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        tree = ast_comments.parse(code)
        updated_code = self.modify_ast(tree)
        return updated_code

    def modify_ast(self, tree):
        if False:
            return 10
        NameUpdater().visit(tree)
        ast_comments.fix_missing_locations(tree)
        ast_comments.increment_lineno(tree, n=1)
        return ast_comments.unparse(tree)

class NameUpdater(ast_comments.NodeTransformer):

    def generic_visit(self, node):
        if False:
            print('Hello World!')
        if isinstance(node, ast_comments.keyword):
            if node.arg == 'space':
                return node
        for (field, old_value) in ast_comments.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast_comments.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast_comments.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast_comments.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def visit_Expr(self, node):
        if False:
            print('Hello World!')
        if hasattr(node.value, 'left') and hasattr(node.value.left, 'id'):
            node.value.left.id = self.check_dict(StrategyUpdater.name_mapping, node.value.left.id)
            self.visit(node.value)
        return node

    @staticmethod
    def check_dict(current_dict: dict, element: str):
        if False:
            i = 10
            return i + 15
        if element in current_dict:
            element = current_dict[element]
        return element

    def visit_arguments(self, node):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(node.args, list):
            for arg in node.args:
                arg.arg = self.check_dict(StrategyUpdater.name_mapping, arg.arg)
        return node

    def visit_Name(self, node):
        if False:
            return 10
        node.id = self.check_dict(StrategyUpdater.name_mapping, node.id)
        return node

    def visit_Import(self, node):
        if False:
            i = 10
            return i + 15
        return node

    def visit_ImportFrom(self, node):
        if False:
            for i in range(10):
                print('nop')
        return node

    def visit_If(self, node: ast_comments.If):
        if False:
            while True:
                i = 10
        for child in ast_comments.iter_child_nodes(node):
            self.visit(child)
        return node

    def visit_FunctionDef(self, node):
        if False:
            print('Hello World!')
        node.name = self.check_dict(StrategyUpdater.function_mapping, node.name)
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node):
        if False:
            i = 10
            return i + 15
        if isinstance(node.value, ast_comments.Name) and node.value.id == 'trade' and (node.attr == 'nr_of_successful_buys'):
            node.attr = 'nr_of_successful_entries'
        return node

    def visit_ClassDef(self, node):
        if False:
            print('Hello World!')
        if any((isinstance(base, ast_comments.Name) and base.id == 'IStrategy' for base in node.bases)):
            has_interface_version = any((isinstance(child, ast_comments.Assign) and isinstance(child.targets[0], ast_comments.Name) and (child.targets[0].id == 'INTERFACE_VERSION') for child in node.body))
            if not has_interface_version:
                node.body.insert(0, ast_comments.parse('INTERFACE_VERSION = 3').body[0])
            else:
                for child in node.body:
                    if isinstance(child, ast_comments.Assign) and isinstance(child.targets[0], ast_comments.Name) and (child.targets[0].id == 'INTERFACE_VERSION'):
                        child.value = ast_comments.parse('3').body[0].value
        self.generic_visit(node)
        return node

    def visit_Subscript(self, node):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(node.slice, ast_comments.Constant):
            if node.slice.value in StrategyUpdater.rename_dict:
                node.slice.value = StrategyUpdater.rename_dict[node.slice.value]
        if hasattr(node.slice, 'elts'):
            self.visit_elts(node.slice.elts)
        if hasattr(node.slice, 'value'):
            if hasattr(node.slice.value, 'elts'):
                self.visit_elts(node.slice.value.elts)
        return node

    def visit_elts(self, elts):
        if False:
            return 10
        if isinstance(elts, list):
            for elt in elts:
                self.visit_elt(elt)
        else:
            self.visit_elt(elts)
        return elts

    def visit_elt(self, elt):
        if False:
            print('Hello World!')
        if isinstance(elt, ast_comments.Constant) and elt.value in StrategyUpdater.rename_dict:
            elt.value = StrategyUpdater.rename_dict[elt.value]
        if hasattr(elt, 'elts'):
            self.visit_elts(elt.elts)
        if hasattr(elt, 'args'):
            if isinstance(elt.args, ast_comments.arguments):
                self.visit_elts(elt.args)
            else:
                for arg in elt.args:
                    self.visit_elts(arg)
        return elt

    def visit_Constant(self, node):
        if False:
            for i in range(10):
                print('nop')
        node.value = self.check_dict(StrategyUpdater.otif_ot_unfilledtimeout, node.value)
        node.value = self.check_dict(StrategyUpdater.name_mapping, node.value)
        return node