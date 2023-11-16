import numpy as np
_module_input_num = {'_key_find': 0, '_key_filter': 1, '_val_desc': 1}
_module_output_type = {'_key_find': 'att', '_key_filter': 'att', '_val_desc': 'ans'}
INVALID_EXPR = 'INVALID_EXPR'

class Assembler:

    def __init__(self, config):
        if False:
            print('Hello World!')
        self.module_names = config.module_names
        for n_s in range(len(self.module_names)):
            if self.module_names[n_s] == '<eos>':
                self.EOS_idx = n_s
                break
        self.name2idx_dict = {name: n_s for (n_s, name) in enumerate(self.module_names)}

    def module_list2tokens(self, module_list, max_len=None):
        if False:
            for i in range(10):
                print('nop')
        layout_tokens = [self.name2idx_dict[name] for name in module_list]
        if max_len is not None:
            if len(module_list) >= max_len:
                raise ValueError('Not enough time steps to add <eos>')
            layout_tokens += [self.EOS_idx] * (max_len - len(module_list))
        return layout_tokens

    def _layout_tokens2str(self, layout_tokens):
        if False:
            return 10
        return ' '.join([self.module_names[idx] for idx in layout_tokens])

    def _invalid_expr(self, layout_tokens, error_str):
        if False:
            return 10
        return {'module': INVALID_EXPR, 'expr_str': self._layout_tokens2str(layout_tokens), 'error': error_str}

    def _assemble_layout_tokens(self, layout_tokens, batch_idx):
        if False:
            return 10
        if not np.any(layout_tokens == self.EOS_idx):
            return self._invalid_expr(layout_tokens, 'cannot find <eos>')
        decoding_stack = []
        for t in range(len(layout_tokens)):
            module_idx = layout_tokens[t]
            if module_idx == self.EOS_idx:
                break
            module_name = self.module_names[module_idx]
            expr = {'module': module_name, 'output_type': _module_output_type[module_name], 'time_idx': t, 'batch_idx': batch_idx}
            input_num = _module_input_num[module_name]
            if len(decoding_stack) < input_num:
                return self._invalid_expr(layout_tokens, 'not enough input for ' + module_name)
            for n_input in range(input_num - 1, -1, -1):
                stack_top = decoding_stack.pop()
                if stack_top['output_type'] != 'att':
                    return self._invalid_expr(layout_tokens, 'input incompatible for ' + module_name)
                expr['input_%d' % n_input] = stack_top
            decoding_stack.append(expr)
        if len(decoding_stack) != 1:
            return self._invalid_expr(layout_tokens, 'final stack size not equal to 1 (%d remains)' % len(decoding_stack))
        result = decoding_stack[0]
        if result['output_type'] != 'ans':
            return self._invalid_expr(layout_tokens, 'result type must be ans, not att')
        return result

    def assemble(self, layout_tokens_batch):
        if False:
            print('Hello World!')
        (_, batch_size) = layout_tokens_batch.shape
        expr_list = [self._assemble_layout_tokens(layout_tokens_batch[:, batch_i], batch_i) for batch_i in range(batch_size)]
        expr_validity = np.array([expr['module'] != INVALID_EXPR for expr in expr_list], np.bool)
        return (expr_list, expr_validity)