"""Utility function copied over from sklearn/base.py
"""
from __future__ import division
from __future__ import print_function
import numpy as np
import six
from joblib.parallel import cpu_count

def _get_n_jobs(n_jobs):
    if False:
        for i in range(10):
            print('nop')
    'Get number of jobs for the computation.\n    See sklearn/utils/__init__.py for more information.\n\n    This function reimplements the logic of joblib to determine the actual\n    number of jobs depending on the cpu count. If -1 all CPUs are used.\n    If 1 is given, no parallel computing code is used at all, which is useful\n    for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.\n    Thus for n_jobs = -2, all CPUs but one are used.\n    Parameters\n    ----------\n    n_jobs : int\n        Number of jobs stated in joblib convention.\n    Returns\n    -------\n    n_jobs : int\n        The actual number of jobs as positive integer.\n    '
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    else:
        return n_jobs

def _partition_estimators(n_estimators, n_jobs):
    if False:
        while True:
            i = 10
    'Private function used to partition estimators between jobs.\n    See sklearn/ensemble/base.py for more information.\n    '
    n_jobs = min(_get_n_jobs(n_jobs), n_estimators)
    n_estimators_per_job = n_estimators // n_jobs * np.ones(n_jobs, dtype=int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)
    return (n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist())

def _pprint(params, offset=0, printer=repr):
    if False:
        for i in range(10):
            print('nop')
    "Pretty print the dictionary 'params'\n\n    See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html\n    and sklearn/base.py for more information.\n\n    :param params: The dictionary to pretty print\n    :type params: dict\n\n    :param offset: The offset in characters to add at the begin of each line.\n    :type offset: int\n\n    :param printer: The function to convert entries to strings, typically\n        the builtin str or repr\n    :type printer: callable\n\n    :return: None\n    "
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for (i, (k, v)) in enumerate(sorted(six.iteritems(params))):
        if type(v) is float:
            this_repr = '%s=%s' % (k, str(v))
        else:
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if this_line_length + len(this_repr) >= 75 or '\n' in this_repr:
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)
    np.set_printoptions(**options)
    lines = ''.join(params_list)
    lines = '\n'.join((l.rstrip(' ') for l in lines.split('\n')))
    return lines