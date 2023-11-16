from pathlib import Path

def main():
    if False:
        print('Hello World!')
    content = create_content()
    with open('g4f/provider/__init__.py', 'w', encoding='utf-8') as f:
        f.write(content)

def create_content():
    if False:
        return 10
    path = Path()
    paths = path.glob('g4f/provider/*.py')
    paths = [p for p in paths if p.name not in ['__init__.py', 'base_provider.py']]
    classnames = [p.stem for p in paths]
    import_lines = [f'from .{name} import {name}' for name in classnames]
    import_content = '\n'.join(import_lines)
    classnames.insert(0, 'BaseProvider')
    all_content = [f'    "{name}"' for name in classnames]
    all_content = ',\n'.join(all_content)
    all_content = f'__all__ = [\n{all_content},\n]'
    return f'from .base_provider import BaseProvider\n{import_content}\n\n\n{all_content}\n'
if __name__ == '__main__':
    main()