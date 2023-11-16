from __future__ import annotations

def test_issue_1272():
    if False:
        return 10
    '\n\n    https://github.com/online-ml/river/issues/1272\n\n    >>> import river\n    >>> from river import forest, metrics\n\n    >>> model = forest.ARFClassifier(metric=metrics.CrossEntropy())\n    >>> model = model.learn_one({"x": 1}, True)\n\n    >>> model = forest.ARFClassifier()\n    >>> model = model.learn_one({"x": 1}, True)\n\n    '