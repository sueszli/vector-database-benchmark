"""Common interactive utility module."""
import logging
_LOGGER = logging.getLogger(__name__)

def is_in_ipython():
    if False:
        for i in range(10):
            print('nop')
    'Determines if current code is executed within an ipython session.'
    try:
        from IPython import get_ipython
        if get_ipython():
            return True
        return False
    except ImportError:
        return False
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        _LOGGER.info('Unexpected error occurred, treated as not in IPython.', exc_info=True)
        return False

def is_in_notebook():
    if False:
        return 10
    'Determines if current code is executed from an ipython notebook.\n\n  If is_in_notebook() is True, then is_in_ipython() must also be True.\n  '
    is_in_notebook = False
    if is_in_ipython():
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            is_in_notebook = True
    return is_in_notebook

def alter_label_if_ipython(transform, pvalueish):
    if False:
        while True:
            i = 10
    'Alters the label to an interactive label with ipython prompt metadata\n  prefixed for the given transform if the given pvalueish belongs to a\n  user-defined pipeline and current code execution is within an ipython kernel.\n  Otherwise, noop.\n\n  A label is either a user-defined or auto-generated str name of a PTransform\n  that is unique within a pipeline. If current environment is_in_ipython(), Beam\n  can implicitly create interactive labels to replace labels of top-level\n  PTransforms to be applied. The label is formatted as:\n  `Cell {prompt}: {original_label}`.\n  '
    if is_in_ipython():
        from apache_beam.runners.interactive import interactive_environment as ie
        ie.current_env().track_user_pipelines()
        from IPython import get_ipython
        prompt = get_ipython().execution_count
        pipeline = _extract_pipeline_of_pvalueish(pvalueish)
        if pipeline and pipeline in ie.current_env().tracked_user_pipelines:
            transform.label = '[{}]: {}'.format(prompt, transform.label)

def _extract_pipeline_of_pvalueish(pvalueish):
    if False:
        return 10
    'Extracts the pipeline that the given pvalueish belongs to.'
    if isinstance(pvalueish, tuple) and len(pvalueish) > 0:
        pvalue = pvalueish[0]
    elif isinstance(pvalueish, dict) and len(pvalueish) > 0:
        pvalue = next(iter(pvalueish.values()))
    else:
        pvalue = pvalueish
    if hasattr(pvalue, 'pipeline'):
        return pvalue.pipeline
    return None