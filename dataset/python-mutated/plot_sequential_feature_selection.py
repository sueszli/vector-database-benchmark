import matplotlib.pyplot as plt

def plot_sequential_feature_selection(metric_dict, figsize=None, kind='std_dev', color='blue', bcolor='steelblue', marker='o', alpha=0.2, ylabel='Performance', confidence_interval=0.95):
    if False:
        i = 10
        return i + 15
    'Plot feature selection results.\n\n    Parameters\n    ----------\n    metric_dict : mlxtend.SequentialFeatureSelector.get_metric_dict() object\n    figsize : tuple (default: None)\n        Height and width of the figure\n    kind : str (default: "std_dev")\n        The kind of error bar or confidence interval in\n        {\'std_dev\', \'std_err\', \'ci\', None}.\n    color : str (default: "blue")\n        Color of the lineplot (accepts any matplotlib color name)\n    bcolor : str (default: "steelblue").\n        Color of the error bars / confidence intervals\n        (accepts any matplotlib color name).\n    marker : str (default: "o")\n        Marker of the line plot\n        (accepts any matplotlib marker name).\n    alpha : float in [0, 1] (default: 0.2)\n        Transparency of the error bars / confidence intervals.\n    ylabel : str (default: "Performance")\n        Y-axis label.\n    confidence_interval : float (default: 0.95)\n        Confidence level if `kind=\'ci\'`.\n\n    Returns\n    ----------\n    fig : matplotlib.pyplot.figure() object\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/plotting/plot_sequential_feature_selection/\n\n    '
    allowed = {'std_dev', 'std_err', 'ci', None}
    if kind not in allowed:
        raise AttributeError('kind not in %s' % allowed)
    if figsize is not None:
        fig = plt.subplots(figsize=figsize)
    else:
        fig = plt.subplots()
    k_feat = sorted(metric_dict.keys())
    avg = [metric_dict[k]['avg_score'] for k in k_feat]
    if kind:
        (upper, lower) = ([], [])
        if kind == 'ci':
            kind = 'ci_bound'
        for k in k_feat:
            upper.append(metric_dict[k]['avg_score'] + metric_dict[k][kind])
            lower.append(metric_dict[k]['avg_score'] - metric_dict[k][kind])
        plt.fill_between(k_feat, upper, lower, alpha=alpha, color=bcolor, lw=1)
        if kind == 'ci_bound':
            kind = 'Confidence Interval (%d%%)' % (confidence_interval * 100)
    plt.plot(k_feat, avg, color=color, marker=marker)
    plt.ylabel(ylabel)
    plt.xlabel('Number of Features')
    feature_min = len(metric_dict[k_feat[0]]['feature_idx'])
    feature_max = len(metric_dict[k_feat[-1]]['feature_idx'])
    plt.xticks(range(feature_min, feature_max + 1), range(feature_min, feature_max + 1))
    return fig