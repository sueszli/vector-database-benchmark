"""Utilities for input validation"""

class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.
    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    Examples
    --------
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.exceptions import NotFittedError
    >>> try:
    ...     LinearSVC().predict([[1, 2], [2, 3], [3, 4]])
    ... except NotFittedError as e:
    ...     print(repr(e))
    ...                        # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    NotFittedError('This LinearSVC instance is not fitted yet',)
    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation.
    """

def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    if False:
        print('Hello World!')
    'Perform is_fitted validation for estimator.\n    Checks if the estimator is fitted by verifying the presence of\n    "all_or_any" of the passed attributes and raises a NotFittedError with the\n    given message.\n    Parameters\n    ----------\n    estimator : estimator instance.\n        estimator instance for which the check is performed.\n    attributes : attribute name(s) given as string or a list/tuple of strings\n        Eg.:\n            ``["coef_", "estimator_", ...], "coef_"``\n    msg : string\n        The default error message is, "This %(name)s instance is not fitted\n        yet. Call \'fit\' with appropriate arguments before using this method."\n        For custom messages if "%(name)s" is present in the message string,\n        it is substituted for the estimator name.\n        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".\n    all_or_any : callable, {all, any}, default all\n        Specify whether all or any of the given attributes must exist.\n    Returns\n    -------\n    None\n    Raises\n    ------\n    NotFittedError\n        If the attributes are not found.\n    '
    if msg is None:
        msg = "This %(name)s instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
    if not hasattr(estimator, 'fit'):
        raise TypeError('%s is not an estimator instance.' % estimator)
    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]
    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})