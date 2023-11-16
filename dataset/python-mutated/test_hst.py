from __future__ import annotations

def test_missing_features():
    if False:
        while True:
            i = 10
    'Checks that HalfSpaceTrees works even if a feature is missing.\n\n    >>> import random\n    >>> from river import anomaly\n    >>> from river import compose\n    >>> from river import datasets\n    >>> from river import metrics\n    >>> from river import preprocessing\n\n    >>> model = compose.Pipeline(\n    ...     preprocessing.MinMaxScaler(),\n    ...     anomaly.HalfSpaceTrees(seed=42)\n    ... )\n\n    >>> auc = metrics.ROCAUC()\n\n    >>> features = list(next(iter(datasets.CreditCard()))[0].keys())\n    >>> random.seed(42)\n\n    >>> for x, y in datasets.CreditCard().take(8000):\n    ...     del x[random.choice(features)]\n    ...     score = model.score_one(x)\n    ...     model = model.learn_one(x, y)\n    ...     auc = auc.update(y, score)\n\n    >>> auc\n    ROCAUC: 88.68%\n\n    '