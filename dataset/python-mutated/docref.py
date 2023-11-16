"""Documentation reference utilities."""
import typing as t
from deepchecks import __version__
__all__ = ['doclink']
links = {'default': {'supported-metrics-by-string': 'https://docs.deepchecks.com/stable/general/guides/metrics_guide.html#list-of-supported-strings', 'supported-prediction-format': 'https://docs.deepchecks.com/stable/tabular/usage_guides/supported_models.html#supported-tasks-and-predictions-format', 'supported-predictions-format-nlp': 'https://docs.deepchecks.com/stable/nlp/usage_guides/supported_tasks.html#supported-labels-and-predictions-format'}}

def doclink(name: str, default_link: t.Optional[str]=None, template: t.Optional[str]=None, package_version: str=__version__) -> str:
    if False:
        i = 10
        return i + 15
    "Get documentation link.\n\n    Parameters\n    ----------\n    name: str\n        the name of the required link as appears in the links' dictionary.\n    default_link: t.Optional[str], default: None\n        default like to use if no link corresponding to name was found.\n    template: t.Optional[str], default: None\n        a string template in which to incorporate the link.\n    package_version: str\n        which version of the docs to use\n\n    Returns\n    -------\n    str\n        The template text incorporated with the relevant link\n    "
    index = links[package_version] if package_version in links else links.get('default') or {}
    link = index.get(name) or default_link
    if link is None:
        return ''
    return link if template is None else template.format(link=link)