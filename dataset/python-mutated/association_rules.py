from itertools import combinations
import numpy as np
import pandas as pd

def association_rules(df, metric='confidence', min_threshold=0.8, support_only=False):
    if False:
        return 10
    'Generates a DataFrame of association rules including the\n    metrics \'score\', \'confidence\', and \'lift\'\n\n    Parameters\n    -----------\n    df : pandas DataFrame\n      pandas DataFrame of frequent itemsets\n      with columns [\'support\', \'itemsets\']\n\n    metric : string (default: \'confidence\')\n      Metric to evaluate if a rule is of interest.\n      **Automatically set to \'support\' if `support_only=True`.**\n      Otherwise, supported metrics are \'support\', \'confidence\', \'lift\',\n      \'leverage\', \'conviction\' and \'zhangs_metric\'\n      These metrics are computed as follows:\n\n      - support(A->C) = support(A+C) [aka \'support\'], range: [0, 1]\n\n      - confidence(A->C) = support(A+C) / support(A), range: [0, 1]\n\n      - lift(A->C) = confidence(A->C) / support(C), range: [0, inf]\n\n      - leverage(A->C) = support(A->C) - support(A)*support(C),\n        range: [-1, 1]\n\n      - conviction = [1 - support(C)] / [1 - confidence(A->C)],\n        range: [0, inf]\n\n      - zhangs_metric(A->C) =\n        leverage(A->C) / max(support(A->C)*(1-support(A)), support(A)*(support(C)-support(A->C)))\n        range: [-1,1]\n\n\n    min_threshold : float (default: 0.8)\n      Minimal threshold for the evaluation metric,\n      via the `metric` parameter,\n      to decide whether a candidate rule is of interest.\n\n    support_only : bool (default: False)\n      Only computes the rule support and fills the other\n      metric columns with NaNs. This is useful if:\n\n      a) the input DataFrame is incomplete, e.g., does\n      not contain support values for all rule antecedents\n      and consequents\n\n      b) you simply want to speed up the computation because\n      you don\'t need the other metrics.\n\n    Returns\n    ----------\n    pandas DataFrame with columns "antecedents" and "consequents"\n      that store itemsets, plus the scoring metric columns:\n      "antecedent support", "consequent support",\n      "support", "confidence", "lift",\n      "leverage", "conviction"\n      of all rules for which\n      metric(rule) >= min_threshold.\n      Each entry in the "antecedents" and "consequents" columns are\n      of type `frozenset`, which is a Python built-in type that\n      behaves similarly to sets except that it is immutable\n      (For more info, see\n      https://docs.python.org/3.6/library/stdtypes.html#frozenset).\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/\n\n    '
    if not df.shape[0]:
        raise ValueError('The input DataFrame `df` containing the frequent itemsets is empty.')
    if not all((col in df.columns for col in ['support', 'itemsets'])):
        raise ValueError("Dataframe needs to contain the                         columns 'support' and 'itemsets'")

    def conviction_helper(sAC, sA, sC):
        if False:
            i = 10
            return i + 15
        confidence = sAC / sA
        conviction = np.empty(confidence.shape, dtype=float)
        if not len(conviction.shape):
            conviction = conviction[np.newaxis]
            confidence = confidence[np.newaxis]
            sAC = sAC[np.newaxis]
            sA = sA[np.newaxis]
            sC = sC[np.newaxis]
        conviction[:] = np.inf
        conviction[confidence < 1.0] = (1.0 - sC[confidence < 1.0]) / (1.0 - confidence[confidence < 1.0])
        return conviction

    def zhangs_metric_helper(sAC, sA, sC):
        if False:
            i = 10
            return i + 15
        denominator = np.maximum(sAC * (1 - sA), sA * (sC - sAC))
        numerator = metric_dict['leverage'](sAC, sA, sC)
        with np.errstate(divide='ignore', invalid='ignore'):
            zhangs_metric = np.where(denominator == 0, 0, numerator / denominator)
        return zhangs_metric
    metric_dict = {'antecedent support': lambda _, sA, __: sA, 'consequent support': lambda _, __, sC: sC, 'support': lambda sAC, _, __: sAC, 'confidence': lambda sAC, sA, _: sAC / sA, 'lift': lambda sAC, sA, sC: metric_dict['confidence'](sAC, sA, sC) / sC, 'leverage': lambda sAC, sA, sC: metric_dict['support'](sAC, sA, sC) - sA * sC, 'conviction': lambda sAC, sA, sC: conviction_helper(sAC, sA, sC), 'zhangs_metric': lambda sAC, sA, sC: zhangs_metric_helper(sAC, sA, sC)}
    columns_ordered = ['antecedent support', 'consequent support', 'support', 'confidence', 'lift', 'leverage', 'conviction', 'zhangs_metric']
    if support_only:
        metric = 'support'
    elif metric not in metric_dict.keys():
        raise ValueError("Metric must be 'confidence' or 'lift', got '{}'".format(metric))
    keys = df['itemsets'].values
    values = df['support'].values
    frozenset_vect = np.vectorize(lambda x: frozenset(x))
    frequent_items_dict = dict(zip(frozenset_vect(keys), values))
    rule_antecedents = []
    rule_consequents = []
    rule_supports = []
    for k in frequent_items_dict.keys():
        sAC = frequent_items_dict[k]
        for idx in range(len(k) - 1, 0, -1):
            for c in combinations(k, r=idx):
                antecedent = frozenset(c)
                consequent = k.difference(antecedent)
                if support_only:
                    sA = None
                    sC = None
                else:
                    try:
                        sA = frequent_items_dict[antecedent]
                        sC = frequent_items_dict[consequent]
                    except KeyError as e:
                        s = str(e) + 'You are likely getting this error because the DataFrame is missing  antecedent and/or consequent  information. You can try using the  `support_only=True` option'
                        raise KeyError(s)
                score = metric_dict[metric](sAC, sA, sC)
                if score >= min_threshold:
                    rule_antecedents.append(antecedent)
                    rule_consequents.append(consequent)
                    rule_supports.append([sAC, sA, sC])
    if not rule_supports:
        return pd.DataFrame(columns=['antecedents', 'consequents'] + columns_ordered)
    else:
        rule_supports = np.array(rule_supports).T.astype(float)
        df_res = pd.DataFrame(data=list(zip(rule_antecedents, rule_consequents)), columns=['antecedents', 'consequents'])
        if support_only:
            sAC = rule_supports[0]
            for m in columns_ordered:
                df_res[m] = np.nan
            df_res['support'] = sAC
        else:
            sAC = rule_supports[0]
            sA = rule_supports[1]
            sC = rule_supports[2]
            for m in columns_ordered:
                df_res[m] = metric_dict[m](sAC, sA, sC)
        return df_res