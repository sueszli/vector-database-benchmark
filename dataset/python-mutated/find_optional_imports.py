"""Utility to check that slow imports are not used in the default path."""
import subprocess
import sys
import qiskit

def _main():
    if False:
        while True:
            i = 10
    optional_imports = ['networkx', 'sympy', 'pydot', 'pygments', 'ipywidgets', 'scipy.stats', 'matplotlib', 'qiskit.providers.aer', 'qiskit.providers.ibmq', 'qiskit.ignis', 'qiskit.aqua', 'docplex']
    modules_imported = []
    for mod in optional_imports:
        if mod in sys.modules:
            modules_imported.append(mod)
    if not modules_imported:
        sys.exit(0)
    res = subprocess.run([sys.executable, '-X', 'importtime', '-c', 'import qiskit'], capture_output=True, encoding='utf8', check=True)
    import_tree = [x.split('|')[-1] for x in res.stderr.split('\n') if 'RuntimeWarning' not in x or 'warnings.warn' not in x]
    indent = -1
    matched_module = None
    for module in import_tree:
        line_indent = len(module) - len(module.lstrip())
        module_name = module.strip()
        if module_name in modules_imported:
            if indent > 0:
                continue
            indent = line_indent
            matched_module = module_name
        if indent > 0:
            if line_indent < indent:
                print(f'ERROR: {matched_module} is imported via {module_name}')
                indent = -1
                matched_module = None
    sys.exit(len(modules_imported))
if __name__ == '__main__':
    _main()