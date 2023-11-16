import numpy
import six
import chainer
from chainer import backend
from chainer import reporter
from chainer.training import extension
from chainer.training import trigger as trigger_module
_default_statistics = {'mean': lambda x: backend.get_array_module(x).mean(x), 'std': lambda x: backend.get_array_module(x).std(x), 'min': lambda x: backend.get_array_module(x).min(x), 'max': lambda x: backend.get_array_module(x).max(x), 'zeros': lambda x: backend.get_array_module(x).count_nonzero(x == 0), 'percentile': lambda x: backend.get_array_module(x).percentile(x, (0.13, 2.28, 15.87, 50, 84.13, 97.72, 99.87))}

class ParameterStatistics(extension.Extension):
    """Trainer extension to report parameter statistics.

    Statistics are collected and reported for a given :class:`~chainer.Link`
    or an iterable of :class:`~chainer.Link`\\ s. If a link contains child
    links, the statistics are reported separately for each child.

    Any function that takes a one-dimensional :class:`numpy.ndarray` or a
    :class:`cupy.ndarray` and outputs a single or multiple real numbers can be
    registered to handle the collection of statistics, e.g.
    :meth:`numpy.ndarray.mean`.

    The keys of reported statistics follow the convention of link name
    followed by parameter name, attribute name and function name, e.g.
    ``VGG16Layers/conv1_1/W/data/mean``. They are prepended with an optional
    prefix and appended with integer indices if the statistics generating
    function return multiple values.

    Args:
        links (~chainer.Link or iterable of ~chainer.Link): Link(s) containing
            the parameters to observe. The link is expected to have a ``name``
            attribute which is used as a part of the report key.
        statistics (dict or 'default'): Dictionary with function name to
            function mappings.
            The name is a string and is used as a part of the report key. The
            function is responsible for generating the statistics.
            If the special value ``'default'`` is specified, the default
            statistics functions will be used.
        report_params (bool): If ``True``, report statistics for parameter
            values such as weights and biases.
        report_grads (bool): If ``True``, report statistics for parameter
            gradients.
        prefix (str): Optional prefix to prepend to the report keys.
        trigger: Trigger that decides when to aggregate the results and report
            the values.
        skip_nan_params (bool): If ``True``, statistics are not computed for
            parameters including NaNs and a single NaN value is immediately
            reported instead. Otherwise, this extension will simply try to
            compute the statistics without performing any checks for NaNs.

    .. note::

       The default statistic functions are as follows:

       * ``'mean'`` (``xp.mean(x)``)
       * ``'std'`` (``xp.std(x)``)
       * ``'min'`` (``xp.min(x)``)
       * ``'max'`` (``xp.max(x)``)
       * ``'zeros'`` (``xp.count_nonzero(x == 0)``)
       * ``'percentile'`` (``xp.percentile(x, (0.13, 2.28, 15.87, 50, 84.13, 97.72, 99.87))``)

    """
    default_name = 'parameter_statistics'
    priority = extension.PRIORITY_WRITER
    report_key_template = '{prefix}{link_name}{param_name}/{attr_name}/{function_name}'
    default_statistics = _default_statistics

    def __init__(self, links, statistics='default', report_params=True, report_grads=True, prefix=None, trigger=(1, 'epoch'), skip_nan_params=False):
        if False:
            i = 10
            return i + 15
        if not isinstance(links, (list, tuple)):
            links = (links,)
        self._links = links
        if statistics is None:
            statistics = {}
        elif statistics == 'default':
            statistics = self.default_statistics
        self._statistics = dict(statistics)
        attrs = []
        if report_params:
            attrs.append('data')
        if report_grads:
            attrs.append('grad')
        self._attrs = attrs
        self._prefix = prefix
        self._trigger = trigger_module.get_trigger(trigger)
        self._summary = reporter.DictSummary()
        self._skip_nan_params = skip_nan_params

    def __call__(self, trainer):
        if False:
            print('Hello World!')
        'Execute the statistics extension.\n\n        Collect statistics for the current state of parameters.\n\n        Note that this method will merely update its statistic summary, unless\n        the internal trigger is fired. If the trigger is fired, the summary\n        will also be reported and then reset for the next accumulation.\n\n        Args:\n            trainer (~chainer.training.Trainer): Associated trainer that\n                invoked this extension.\n        '
        statistics = {}
        for link in self._links:
            link_name = getattr(link, 'name', 'None')
            for (param_name, param) in link.namedparams():
                for attr_name in self._attrs:
                    for (function_name, function) in six.iteritems(self._statistics):
                        params = getattr(param, attr_name).ravel()
                        if self._skip_nan_params and backend.get_array_module(params).isnan(params).any():
                            value = numpy.nan
                        else:
                            value = function(params)
                        key = self.report_key_template.format(prefix=self._prefix + '/' if self._prefix else '', link_name=link_name, param_name=param_name, attr_name=attr_name, function_name=function_name)
                        if isinstance(value, chainer.get_array_types()) and value.size > 1:
                            statistics.update({'{}/{}'.format(key, i): v for (i, v) in enumerate(value)})
                        else:
                            statistics[key] = value
        self._summary.add(statistics)
        if self._trigger(trainer):
            reporter.report(self._summary.compute_mean())
            self._summary = reporter.DictSummary()

    def register_statistics(self, name, function):
        if False:
            return 10
        'Register a function to compute a certain statistic.\n\n        The registered function will be called each time the extension runs and\n        the results will be included in the report.\n\n        Args:\n            name (str): Name of the statistic.\n            function: Function to generate the statistic. Any function that\n                takes a one-dimensional :class:`numpy.ndarray` or a\n                :class:`cupy.ndarray` and outputs a single or multiple real\n                numbers is allowed.\n        '
        self._statistics[name] = function