import collections
from paddle.utils import gast
from .base_transformer import BaseTransformer

class RegisterHookTransformer(BaseTransformer):

    def __init__(self, root):
        if False:
            for i in range(10):
                print('nop')
        self.register_hook_pos_map = collections.defaultdict(list)
        self.assignment_pos_map = collections.defaultdict(list)
        self.root = root

    def transform(self):
        if False:
            i = 10
            return i + 15
        '\n        Main function to transform AST.\n        '
        self.visit(self.root)

    def visit_FunctionDef(self, func_def):
        if False:
            for i in range(10):
                print('nop')
        check_register_hook = next((node for node in gast.walk(func_def) if isinstance(node, gast.Attribute) and node.attr == 'register_hook'), None)
        if check_register_hook is None:
            return func_def
        register_hook_pos_map = self.register_hook_pos_map
        assignment_pos_map = self.assignment_pos_map
        for i in range(len(func_def.body) - 1, -1, -1):
            body = func_def.body[i]
            if isinstance(body, gast.Expr):
                for node in gast.walk(body):
                    if isinstance(node, gast.Attribute) and node.attr == 'register_hook':
                        param_name = node.value.id
                        register_hook_pos_map[param_name].append(i)
            elif isinstance(body, gast.Assign):
                for target in body.targets:
                    assignment_pos_map[target.id].append(i)
        order_map = {}
        for (k, idx_list) in register_hook_pos_map.items():
            for idx in idx_list:
                if k not in assignment_pos_map:
                    order_map[idx] = 1
                else:
                    for assignment_idx in assignment_pos_map[k]:
                        if idx > assignment_idx:
                            order_map[idx] = assignment_idx + 1
                            break
        code_order = [*range(len(func_def.body))]
        for (k, v) in sorted(order_map.items(), key=lambda x: x[1], reverse=True):
            if k == v:
                continue
            code_order.remove(k)
            code_order.insert(v, k)
        new_body = [func_def.body[i] for i in code_order]
        func_def.body = new_body
        return func_def