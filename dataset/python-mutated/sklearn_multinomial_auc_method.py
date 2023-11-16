import numpy as np
from itertools import combinations
from functools import partial
from sklearn.utils import column_or_1d, check_consistent_length, check_array
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc, roc_curve
from sklearn.utils.multiclass import type_of_target

def _encode_check_unknown(values, uniques, return_mask=False):
    if False:
        print('Hello World!')
    '\n    Helper function to check for unknowns in values to be encoded.\n\n    Uses pure python method for object dtype, and numpy method for\n    all other dtypes.\n\n    Parameters\n    ----------\n    values : array\n        Values to check for unknowns.\n    uniques : array\n        Allowed uniques values.\n    return_mask : bool, default False\n        If True, return a mask of the same shape as `values` indicating\n        the valid values.\n\n    Returns\n    -------\n    diff : list\n        The unique values present in `values` and not in `uniques` (the\n        unknown values).\n    valid_mask : boolean array\n        Additionally returned if ``return_mask=True``.\n\n    '
    if values.dtype == object:
        uniques_set = set(uniques)
        diff = list(set(values) - uniques_set)
        if return_mask:
            if diff:
                valid_mask = np.array([val in uniques_set for val in values])
            else:
                valid_mask = np.ones(len(values), dtype=bool)
            return (diff, valid_mask)
        else:
            return diff
    else:
        unique_values = np.unique(values)
        diff = list(np.setdiff1d(unique_values, uniques, assume_unique=True))
        if return_mask:
            if diff:
                valid_mask = np.in1d(values, uniques)
            else:
                valid_mask = np.ones(len(values), dtype=bool)
            return (diff, valid_mask)
        else:
            return diff

def _encode_numpy(values, uniques=None, encode=False, check_unknown=True):
    if False:
        i = 10
        return i + 15
    if uniques is None:
        if encode:
            (uniques, encoded) = np.unique(values, return_inverse=True)
            return (uniques, encoded)
        else:
            return np.unique(values)
    if encode:
        if check_unknown:
            diff = _encode_check_unknown(values, uniques)
            if diff:
                raise ValueError('y contains previously unseen labels: %s'.format(str(diff)))
        encoded = np.searchsorted(uniques, values)
        return (uniques, encoded)
    else:
        return uniques

def _encode_python(values, uniques=None, encode=False):
    if False:
        while True:
            i = 10
    if uniques is None:
        uniques = sorted(set(values))
        uniques = np.array(uniques, dtype=values.dtype)
    if encode:
        table = {val: i for (i, val) in enumerate(uniques)}
        try:
            encoded = np.array([table[v] for v in values])
        except KeyError as e:
            raise ValueError('y contains previously unseen labels: %s'.format(str(e)))
        return (uniques, encoded)
    else:
        return uniques

def _encode(values, uniques=None, encode=False, check_unknown=True):
    if False:
        return 10
    'Helper function to factorize (find uniques) and encode values.\n\n    Uses pure python method for object dtype, and numpy method for\n    all other dtypes.\n    The numpy method has the limitation that the `uniques` need to\n    be sorted. Importantly, this is not checked but assumed to already be\n    the case. The calling method needs to ensure this for all non-object\n    values.\n\n    Parameters\n    ----------\n    values : array\n        Values to factorize or encode.\n    uniques : array, optional\n        If passed, uniques are not determined from passed values (this\n        can be because the user specified categories, or because they\n        already have been determined in fit).\n    encode : bool, default False\n        If True, also encode the values into integer codes based on `uniques`.\n    check_unknown : bool, default True\n        If True, check for values in ``values`` that are not in ``unique``\n        and raise an error. This is ignored for object dtype, and treated as\n        True in this case. This parameter is useful for\n        _BaseEncoder._transform() to avoid calling _encode_check_unknown()\n        twice.\n\n    Returns\n    -------\n    uniques\n        If ``encode=False``. The unique values are sorted if the `uniques`\n        parameter was None (and thus inferred from the data).\n    (uniques, encoded)\n        If ``encode=True``.\n\n    '
    if values.dtype == object:
        try:
            res = _encode_python(values, uniques, encode)
        except TypeError:
            types = sorted((t.__qualname__ for t in set((type(v) for v in values))))
            raise TypeError('Encoders require their input to be uniformly ' + 'strings or numbers. Got ' + ' '.join(types))
        return res
    else:
        return _encode_numpy(values, uniques, encode, check_unknown=check_unknown)

def roc_auc_score(y_true, y_score, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None):
    if False:
        for i in range(10):
            print('nop')
    "Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)\n    from prediction scores.\n    Note: this implementation can be used with binary, multiclass and\n    multilabel classification, but some restrictions apply (see Parameters).\n    Read more in the :ref:`User Guide <roc_metrics>`.\n    Parameters\n    ----------\n    y_true : array-like of shape (n_samples,) or (n_samples, n_classes)\n        True labels or binary label indicators. The binary and multiclass cases\n        expect labels with shape (n_samples,) while the multilabel case expects\n        binary label indicators with shape (n_samples, n_classes).\n    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)\n        Target scores. In the binary and multilabel cases, these can be either\n        probability estimates or non-thresholded decision values (as returned\n        by `decision_function` on some classifiers). In the multiclass case,\n        these must be probability estimates which sum to 1. The binary\n        case expects a shape (n_samples,), and the scores must be the scores of\n        the class with the greater label. The multiclass and multilabel\n        cases expect a shape (n_samples, n_classes). In the multiclass case,\n        the order of the class scores must correspond to the order of\n        ``labels``, if provided, or else to the numerical or lexicographical\n        order of the labels in ``y_true``.\n    average : {'micro', 'macro', 'samples', 'weighted'} or None,             default='macro'\n        If ``None``, the scores for each class are returned. Otherwise,\n        this determines the type of averaging performed on the data:\n        Note: multiclass ROC AUC currently only handles the 'macro' and\n        'weighted' averages.\n        ``'micro'``:\n            Calculate metrics globally by considering each element of the label\n            indicator matrix as a label.\n        ``'macro'``:\n            Calculate metrics for each label, and find their unweighted\n            mean.  This does not take label imbalance into account.\n        ``'weighted'``:\n            Calculate metrics for each label, and find their average, weighted\n            by support (the number of true instances for each label).\n        ``'samples'``:\n            Calculate metrics for each instance, and find their average.\n        Will be ignored when ``y_true`` is binary.\n    sample_weight : array-like of shape (n_samples,), default=None\n        Sample weights.\n    max_fpr : float > 0 and <= 1, default=None\n        If not ``None``, the standardized partial AUC [2]_ over the range\n        [0, max_fpr] is returned. For the multiclass case, ``max_fpr``,\n        should be either equal to ``None`` or ``1.0`` as AUC ROC partial\n        computation currently is not supported for multiclass.\n    multi_class : {'raise', 'ovr', 'ovo'}, default='raise'\n        Multiclass only. Determines the type of configuration to use. The\n        default value raises an error, so either ``'ovr'`` or ``'ovo'`` must be\n        passed explicitly.\n        ``'ovr'``:\n            Computes the AUC of each class against the rest [3]_ [4]_. This\n            treats the multiclass case in the same way as the multilabel case.\n            Sensitive to class imbalance even when ``average == 'macro'``,\n            because class imbalance affects the composition of each of the\n            'rest' groupings.\n        ``'ovo'``:\n            Computes the average AUC of all possible pairwise combinations of\n            classes [5]_. Insensitive to class imbalance when\n            ``average == 'macro'``.\n    labels : array-like of shape (n_classes,), default=None\n        Multiclass only. List of labels that index the classes in ``y_score``.\n        If ``None``, the numerical or lexicographical order of the labels in\n        ``y_true`` is used.\n    Returns\n    -------\n    auc : float\n    References\n    ----------\n    .. [1] `Wikipedia entry for the Receiver operating characteristic\n            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_\n    .. [2] `Analyzing a portion of the ROC curve. McClish, 1989\n            <https://www.ncbi.nlm.nih.gov/pubmed/2668680>`_\n    .. [3] Provost, F., Domingos, P. (2000). Well-trained PETs: Improving\n           probability estimation trees (Section 6.2), CeDER Working Paper\n           #IS-00-04, Stern School of Business, New York University.\n    .. [4] `Fawcett, T. (2006). An introduction to ROC analysis. Pattern\n            Recognition Letters, 27(8), 861-874.\n            <https://www.sciencedirect.com/science/article/pii/S016786550500303X>`_\n    .. [5] `Hand, D.J., Till, R.J. (2001). A Simple Generalisation of the Area\n            Under the ROC Curve for Multiple Class Classification Problems.\n            Machine Learning, 45(2), 171-186.\n            <http://link.springer.com/article/10.1023/A:1010920819831>`_\n    See also\n    --------\n    average_precision_score : Area under the precision-recall curve\n    roc_curve : Compute Receiver operating characteristic (ROC) curve\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from sklearn.metrics import roc_auc_score\n    >>> y_true = np.array([0, 0, 1, 1])\n    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])\n    >>> roc_auc_score(y_true, y_scores)\n    0.75\n    "
    y_type = type_of_target(y_true)
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_score = check_array(y_score, ensure_2d=False)
    if y_type == 'multiclass' or (y_type == 'binary' and y_score.ndim == 2 and (y_score.shape[1] > 2)):
        if max_fpr is not None and max_fpr != 1.0:
            raise ValueError("Partial AUC computation not available in multiclass setting, 'max_fpr' must be set to `None`, received `max_fpr={0}` instead".format(max_fpr))
        if multi_class == 'raise':
            raise ValueError("multi_class must be in ('ovo', 'ovr')")
        return _multiclass_roc_auc_score(y_true, y_score, labels, multi_class, average, sample_weight)
    elif y_type == 'binary':
        labels = np.unique(y_true)
        y_true = label_binarize(y_true, classes=labels)[:, 0]
        return _average_binary_score(partial(_binary_roc_auc_score, max_fpr=max_fpr), y_true, y_score, average, sample_weight=sample_weight)
    else:
        return _average_binary_score(partial(_binary_roc_auc_score, max_fpr=max_fpr), y_true, y_score, average, sample_weight=sample_weight)

def _average_binary_score(binary_metric, y_true, y_score, average, sample_weight=None):
    if False:
        i = 10
        return i + 15
    "Average a binary metric for multilabel classification\n\n    Parameters\n    ----------\n    y_true : array, shape = [n_samples] or [n_samples, n_classes]\n        True binary labels in binary label indicators.\n\n    y_score : array, shape = [n_samples] or [n_samples, n_classes]\n        Target scores, can either be probability estimates of the positive\n        class, confidence values, or binary decisions.\n\n    average : string, [None, 'micro', 'macro' (default), 'samples', 'weighted']\n        If ``None``, the scores for each class are returned. Otherwise,\n        this determines the type of averaging performed on the data:\n\n        ``'micro'``:\n            Calculate metrics globally by considering each element of the label\n            indicator matrix as a label.\n        ``'macro'``:\n            Calculate metrics for each label, and find their unweighted\n            mean.  This does not take label imbalance into account.\n        ``'weighted'``:\n            Calculate metrics for each label, and find their average, weighted\n            by support (the number of true instances for each label).\n        ``'samples'``:\n            Calculate metrics for each instance, and find their average.\n\n        Will be ignored when ``y_true`` is binary.\n\n    sample_weight : array-like of shape (n_samples,), default=None\n        Sample weights.\n\n    binary_metric : callable, returns shape [n_classes]\n        The binary metric function to use.\n\n    Returns\n    -------\n    score : float or array of shape [n_classes]\n        If not ``None``, average the score, else return the score for each\n        classes.\n\n    "
    average_options = (None, 'micro', 'macro', 'weighted', 'samples')
    if average not in average_options:
        raise ValueError('average has to be one of {0}'.format(average_options))
    y_type = type_of_target(y_true)
    if y_type not in ('binary', 'multilabel-indicator'):
        raise ValueError('{0} format is not supported'.format(y_type))
    if y_type == 'binary':
        return binary_metric(y_true, y_score, sample_weight=sample_weight)
    check_consistent_length(y_true, y_score, sample_weight)
    y_true = check_array(y_true)
    y_score = check_array(y_score)
    not_average_axis = 1
    score_weight = sample_weight
    average_weight = None
    if average == 'micro':
        if score_weight is not None:
            score_weight = np.repeat(score_weight, y_true.shape[1])
        y_true = y_true.ravel()
        y_score = y_score.ravel()
    elif average == 'weighted':
        if score_weight is not None:
            average_weight = np.sum(np.multiply(y_true, np.reshape(score_weight, (-1, 1))), axis=0)
        else:
            average_weight = np.sum(y_true, axis=0)
        if np.isclose(average_weight.sum(), 0.0):
            return 0
    elif average == 'samples':
        average_weight = score_weight
        score_weight = None
        not_average_axis = 0
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_score.ndim == 1:
        y_score = y_score.reshape((-1, 1))
    n_classes = y_score.shape[not_average_axis]
    score = np.zeros((n_classes,))
    for c in range(n_classes):
        y_true_c = y_true.take([c], axis=not_average_axis).ravel()
        y_score_c = y_score.take([c], axis=not_average_axis).ravel()
        score[c] = binary_metric(y_true_c, y_score_c, sample_weight=score_weight)
    if average is not None:
        if average_weight is not None:
            average_weight = np.asarray(average_weight)
            score[average_weight == 0] = 0
        return np.average(score, weights=average_weight)
    else:
        return score

def _average_multiclass_ovo_score(binary_metric, y_true, y_score, average='macro'):
    if False:
        return 10
    "Average one-versus-one scores for multiclass classification.\n    Uses the binary metric for one-vs-one multiclass classification,\n    where the score is computed according to the Hand & Till (2001) algorithm.\n    Parameters\n    ----------\n    binary_metric : callable\n        The binary metric function to use that accepts the following as input:\n            y_true_target : array, shape = [n_samples_target]\n                Some sub-array of y_true for a pair of classes designated\n                positive and negative in the one-vs-one scheme.\n            y_score_target : array, shape = [n_samples_target]\n                Scores corresponding to the probability estimates\n                of a sample belonging to the designated positive class label\n    y_true : array-like of shape (n_samples,)\n        True multiclass labels.\n    y_score : array-like of shape (n_samples, n_classes)\n        Target scores corresponding to probability estimates of a sample\n        belonging to a particular class.\n    average : {'macro', 'weighted'}, default='macro'\n        Determines the type of averaging performed on the pairwise binary\n        metric scores:\n        ``'macro'``:\n            Calculate metrics for each label, and find their unweighted\n            mean. This does not take label imbalance into account. Classes\n            are assumed to be uniformly distributed.\n        ``'weighted'``:\n            Calculate metrics for each label, taking into account the\n            prevalence of the classes.\n    Returns\n    -------\n    score : float\n        Average of the pairwise binary metric scores.\n    "
    check_consistent_length(y_true, y_score)
    y_true_unique = np.unique(y_true)
    n_classes = y_true_unique.shape[0]
    n_pairs = n_classes * (n_classes - 1) // 2
    pair_scores = np.empty(n_pairs)
    is_weighted = average == 'weighted'
    prevalence = np.empty(n_pairs) if is_weighted else None
    for (ix, (a, b)) in enumerate(combinations(y_true_unique, 2)):
        a_mask = y_true == a
        b_mask = y_true == b
        ab_mask = np.logical_or(a_mask, b_mask)
        if is_weighted:
            prevalence[ix] = np.average(ab_mask)
        a_true = a_mask[ab_mask]
        b_true = b_mask[ab_mask]
        a_true_score = binary_metric(a_true, y_score[ab_mask, a])
        b_true_score = binary_metric(b_true, y_score[ab_mask, b])
        pair_scores[ix] = (a_true_score + b_true_score) / 2
    return np.average(pair_scores, weights=prevalence)

def _multiclass_roc_auc_score(y_true, y_score, labels, multi_class, average, sample_weight):
    if False:
        i = 10
        return i + 15
    "Multiclass roc auc score\n    Parameters\n    ----------\n    y_true : array-like of shape (n_samples,)\n        True multiclass labels.\n    y_score : array-like of shape (n_samples, n_classes)\n        Target scores corresponding to probability estimates of a sample\n        belonging to a particular class\n    labels : array, shape = [n_classes] or None, optional (default=None)\n        List of labels to index ``y_score`` used for multiclass. If ``None``,\n        the lexical order of ``y_true`` is used to index ``y_score``.\n    multi_class : string, 'ovr' or 'ovo'\n        Determines the type of multiclass configuration to use.\n        ``'ovr'``:\n            Calculate metrics for the multiclass case using the one-vs-rest\n            approach.\n        ``'ovo'``:\n            Calculate metrics for the multiclass case using the one-vs-one\n            approach.\n    average : 'macro' or 'weighted', optional (default='macro')\n        Determines the type of averaging performed on the pairwise binary\n        metric scores\n        ``'macro'``:\n            Calculate metrics for each label, and find their unweighted\n            mean. This does not take label imbalance into account. Classes\n            are assumed to be uniformly distributed.\n        ``'weighted'``:\n            Calculate metrics for each label, taking into account the\n            prevalence of the classes.\n    sample_weight : array-like of shape (n_samples,), default=None\n        Sample weights.\n    "
    if not np.allclose(1, y_score.sum(axis=1)):
        raise ValueError('Target scores need to be probabilities for multiclass roc_auc, i.e. they should sum up to 1.0 over classes')
    average_options = ('macro', 'weighted')
    if average not in average_options:
        raise ValueError('average must be one of {0} for multiclass problems'.format(average_options))
    multiclass_options = ('ovo', 'ovr')
    if multi_class not in multiclass_options:
        raise ValueError("multi_class='{0}' is not supported for multiclass ROC AUC, multi_class must be in {1}".format(multi_class, multiclass_options))
    if labels is not None:
        labels = column_or_1d(labels)
        classes = _encode(labels)
        if len(classes) != len(labels):
            raise ValueError("Parameter 'labels' must be unique")
        if not np.array_equal(classes, labels):
            raise ValueError("Parameter 'labels' must be ordered")
        if len(classes) != y_score.shape[1]:
            raise ValueError("Number of given labels, {0}, not equal to the number of columns in 'y_score', {1}".format(len(classes), y_score.shape[1]))
        if len(np.setdiff1d(y_true, classes)):
            raise ValueError("'y_true' contains labels not in parameter 'labels'")
    else:
        classes = _encode(y_true)
        if len(classes) != y_score.shape[1]:
            raise ValueError("Number of classes in y_true not equal to the number of columns in 'y_score'")
    if multi_class == 'ovo':
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported for multiclass one-vs-one ROC AUC, 'sample_weight' must be None in this case.")
        (_, y_true_encoded) = _encode(y_true, uniques=classes, encode=True)
        return _average_multiclass_ovo_score(_binary_roc_auc_score, y_true_encoded, y_score, average=average)
    else:
        y_true_multilabel = label_binarize(y_true, classes=classes)
        return _average_binary_score(_binary_roc_auc_score, y_true_multilabel, y_score, average, sample_weight=sample_weight)

def _binary_roc_auc_score(y_true, y_score, sample_weight=None, max_fpr=None):
    if False:
        for i in range(10):
            print('nop')
    'Binary roc auc score'
    if len(np.unique(y_true)) != 2:
        raise ValueError('Only one class present in y_true. ROC AUC score is not defined in that case.')
    (fpr, tpr, _) = roc_curve(y_true, y_score, sample_weight=sample_weight)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError('Expected max_fpr in range (0, 1], got: %r'.format(max_fpr))
    stop = np.searchsorted(fpr, max_fpr, 'right')
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)
    min_area = 0.5 * max_fpr ** 2
    max_area = max_fpr
    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))