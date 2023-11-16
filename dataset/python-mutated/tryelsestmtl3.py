from uncompyle6.parsers.treenode import SyntaxTree

def tryelsestmtl3(self, lhs, n, rule, ast, tokens, first, last):
    if False:
        for i in range(10):
            print('nop')
    except_handler = ast[3]
    if except_handler == 'except_handler_else':
        except_handler = except_handler[0]
    come_from = except_handler[-1]
    if come_from == 'COME_FROM':
        first_come_from = except_handler[-1]
    elif come_from == 'END_FINALLY':
        return False
    elif come_from == 'except_return':
        return False
    else:
        assert come_from in ('come_froms', 'opt_come_from_except')
        first_come_from = come_from[0]
        if not hasattr(first_come_from, 'attr'):
            return False
    leading_jump = except_handler[0]
    if not hasattr(leading_jump, 'offset'):
        return False
    if isinstance(leading_jump, SyntaxTree):
        except_handler_first_offset = leading_jump.first_child().off2int()
    else:
        except_handler_first_offset = leading_jump.off2int()
    return first_come_from.attr > except_handler_first_offset