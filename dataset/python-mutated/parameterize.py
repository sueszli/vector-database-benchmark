import nbformat
from .engines import papermill_engines
from .log import logger
from .exceptions import PapermillMissingParameterException
from .iorw import read_yaml_file
from .translators import translate_parameters
from .utils import find_first_tagged_cell_index
from uuid import uuid4
from datetime import datetime

def add_builtin_parameters(parameters):
    if False:
        return 10
    'Add built-in parameters to a dictionary of parameters\n\n    Parameters\n    ----------\n    parameters : dict\n       Dictionary of parameters provided by the user\n    '
    with_builtin_parameters = {'pm': {'run_uuid': str(uuid4()), 'current_datetime_local': datetime.now(), 'current_datetime_utc': datetime.utcnow()}}
    if parameters is not None:
        with_builtin_parameters.update(parameters)
    return with_builtin_parameters

def parameterize_path(path, parameters):
    if False:
        print('Hello World!')
    'Format a path with a provided dictionary of parameters\n\n    Parameters\n    ----------\n    path : string or nbformat.NotebookNode or None\n       Path with optional parameters, as a python format string. If path is a NotebookNode\n       or None, the path is returned without modification\n    parameters : dict or None\n       Arbitrary keyword arguments to fill in the path\n    '
    if path is None or isinstance(path, nbformat.NotebookNode):
        return path
    if parameters is None:
        parameters = {}
    try:
        return path.format(**parameters)
    except KeyError as key_error:
        raise PapermillMissingParameterException(f'Missing parameter {key_error}')

def parameterize_notebook(nb, parameters, report_mode=False, comment='Parameters', kernel_name=None, language=None, engine_name=None):
    if False:
        while True:
            i = 10
    'Assigned parameters into the appropriate place in the input notebook\n\n    Parameters\n    ----------\n    nb : NotebookNode\n       Executable notebook object\n    parameters : dict\n       Arbitrary keyword arguments to pass as notebook parameters\n    report_mode : bool, optional\n       Flag to set report mode\n    comment : str, optional\n        Comment added to the injected cell\n    '
    if isinstance(parameters, str):
        parameters = read_yaml_file(parameters)
    kernel_name = papermill_engines.nb_kernel_name(engine_name, nb, kernel_name)
    language = papermill_engines.nb_language(engine_name, nb, language)
    param_content = translate_parameters(kernel_name, language, parameters, comment)
    nb = nbformat.v4.upgrade(nb)
    newcell = nbformat.v4.new_code_cell(source=param_content)
    newcell.metadata['tags'] = ['injected-parameters']
    if report_mode:
        newcell.metadata['jupyter'] = newcell.get('jupyter', {})
        newcell.metadata['jupyter']['source_hidden'] = True
    param_cell_index = find_first_tagged_cell_index(nb, 'parameters')
    injected_cell_index = find_first_tagged_cell_index(nb, 'injected-parameters')
    if injected_cell_index >= 0:
        before = nb.cells[:injected_cell_index]
        after = nb.cells[injected_cell_index + 1:]
    elif param_cell_index >= 0:
        before = nb.cells[:param_cell_index + 1]
        after = nb.cells[param_cell_index + 1:]
    else:
        logger.warning("Input notebook does not contain a cell with tag 'parameters'")
        before = []
        after = nb.cells
    nb.cells = before + [newcell] + after
    nb.metadata.papermill['parameters'] = parameters
    return nb