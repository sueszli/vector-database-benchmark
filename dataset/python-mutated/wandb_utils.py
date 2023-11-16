"""Wandb utils module."""
import contextlib
import typing as t
__all__ = ['wandb_run']
WANDB_INSTALLATION_CMD = 'pip install wandb'

@contextlib.contextmanager
def wandb_run(project: t.Optional[str]=None, **kwargs) -> t.Iterator[t.Any]:
    if False:
        i = 10
        return i + 15
    "Create new one or use existing wandb run instance.\n\n    Parameters\n    ----------\n    project : Optional[str], default None\n        project name\n    **kwargs :\n        additional parameters that will be passed to the 'wandb.init'\n\n    Returns\n    -------\n    Iterator[wandb.sdk.wandb_run.Run]\n    "
    try:
        import wandb
    except ImportError as error:
        raise ImportError(f'"wandb_run" requires the wandb python package. To get it, run - {WANDB_INSTALLATION_CMD}.') from error
    else:
        if wandb.run is not None:
            yield wandb.run
        else:
            kwargs = {'project': project or 'deepchecks', **kwargs}
            with t.cast(t.ContextManager, wandb.init(**kwargs)) as run:
                wandb.run._label(repo='Deepchecks')
                yield run