from sympy.printing.pycode import PythonCodePrinter
' This module collects utilities for rendering Python code. '

def render_as_module(content, standard='python3'):
    if False:
        for i in range(10):
            print('nop')
    'Renders Python code as a module (with the required imports).\n\n    Parameters\n    ==========\n\n    standard :\n        See the parameter ``standard`` in\n        :meth:`sympy.printing.pycode.pycode`\n    '
    printer = PythonCodePrinter({'standard': standard})
    pystr = printer.doprint(content)
    if printer._settings['fully_qualified_modules']:
        module_imports_str = '\n'.join(('import %s' % k for k in printer.module_imports))
    else:
        module_imports_str = '\n'.join(['from %s import %s' % (k, ', '.join(v)) for (k, v) in printer.module_imports.items()])
    return module_imports_str + '\n\n' + pystr