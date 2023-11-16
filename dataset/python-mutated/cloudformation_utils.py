import os
import pathlib
import jinja2
from localstack.utils.common import load_file

def load_template_file(file_path: str | os.PathLike, *, path_ctx: str | os.PathLike=None) -> str:
    if False:
        return 10
    '\n    Load a cloudformation file (YAML or JSON)\n\n    Note this now requires providing the file_path as a proper structured object.\n    In turn this makes it easier to find proper templates by using IDE autocomplete features when selecting templates\n\n    :param file_path: path to file\n    :param path_ctx: *must* be provided if file_path is not an absolute path\n\n    :returns default encoded string representation of file contents\n    '
    file_path_obj = pathlib.Path(file_path)
    if file_path_obj.suffix not in ['.yaml', '.yml', '.json']:
        raise ValueError('Unsupported suffix for template file')
    if path_ctx is not None:
        file_path_obj = file_path_obj.relative_to(path_ctx)
    elif not file_path_obj.is_absolute():
        raise ValueError('Provided path must be absolute if no path_ctx is provided')
    return load_file(file_path_obj.absolute())

def render_template(template_body: str, **template_vars) -> str:
    if False:
        print('Hello World!')
    'render a template with jinja'
    if template_vars:
        template_body = jinja2.Template(template_body).render(**template_vars)
    return template_body

def load_template_raw(path: str):
    if False:
        while True:
            i = 10
    return load_template_file(path)