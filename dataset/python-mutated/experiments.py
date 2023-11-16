from typing import Dict, List, Optional, Union
from rich.text import Text
from dvc.repo import Repo
from dvc.repo.experiments.show import tabulate

def exp_save(name: Optional[str]=None, force: bool=False, include_untracked: Optional[List[str]]=None):
    if False:
        i = 10
        return i + 15
    '\n    Create a new DVC experiment using `exp save`.\n\n    See https://dvc.org/doc/command-reference/exp/save.\n\n    Args:\n        name (str, optional): specify a name for this experiment.\n            If `None`, a default one will be generated, such as `urban-sign`.\n            Defaults to `None`.\n        force (bool):  overwrite the experiment if an experiment with the same\n            name already exists.\n            Defaults to `False`.\n        include_untracked (List[str], optional): specify untracked file(s) to\n            be included in the saved experiment.\n            Defaults to `None`.\n\n    Returns:\n        str: The `Git revision`_ of the created experiment.\n\n    Raises:\n        ExperimentExistsError: If an experiment with `name` already exists and\n            `force=False`.\n\n    .. _Git revision:\n        https://git-scm.com/docs/revisions\n    '
    with Repo() as repo:
        return repo.experiments.save(name=name, force=force, include_untracked=include_untracked)

def _postprocess(exp_rows):
    if False:
        i = 10
        return i + 15
    for exp_row in exp_rows:
        for (k, v) in exp_row.items():
            if isinstance(v, Text):
                v_str = str(v)
                try:
                    exp_row[k] = float(v_str)
                except ValueError:
                    exp_row[k] = v_str
            elif not exp_row[k]:
                exp_row[k] = None
    return exp_rows

def exp_show(repo: Optional[str]=None, revs: Optional[Union[str, List[str]]]=None, num: int=1, param_deps: bool=False, force: bool=False, config: Optional[Dict]=None) -> List[Dict]:
    if False:
        print('Hello World!')
    'Get DVC experiments tracked in `repo`.\n\n    Without arguments, this function will retrieve all experiments derived from\n    the Git `HEAD`.\n\n    See the options below to customize the experiments retrieved.\n\n    Args:\n        repo (str, optional): location of the DVC repository.\n            Defaults to the current project (found by walking up from the\n            current working directory tree).\n            It can be a URL or a file system path.\n            Both HTTP and SSH protocols are supported for online Git repos\n            (e.g. [user@]server:project.git).\n        revs (Union[str, List[str]], optional): Git revision(s) (e.g. branch,\n            tag, SHA commit) to use as a reference point to start listing\n            experiments.\n            Defaults to `None`, which will use `HEAD` as starting point.\n        num (int, optional): show experiments from the last `num` commits\n            (first parents) starting from the `revs` baseline.\n            Give a negative value to include all first-parent commits (similar\n            to `git log -n`).\n            Defaults to 1.\n        param_deps (bool, optional): include only parameters that are stage\n            dependencies.\n            Defaults to `False`.\n        force (bool, optional): force re-collection of experiments instead of\n            loading from internal experiments cache.\n            DVC caches `exp_show` data for completed experiments to improve\n            performance of subsequent calls.\n            When `force` is specified, DVC will reload all experiment data and\n            ignore any previously cached results.\n            Defaults to `False`.\n        config (dict, optional): config to be passed through to DVC project.\n            Defaults to `None`.\n\n    Returns:\n        List[Dict]: Each item in the list will contain a dictionary with\n            the info for an individual experiment.\n            See Examples below.\n    '
    with Repo.open(repo, config=config) as _repo:
        experiments = _repo.experiments.show(revs=revs, num=num, param_deps=param_deps, force=force)
        (td, _) = tabulate(experiments, fill_value=None)
        return _postprocess(td.as_dict())