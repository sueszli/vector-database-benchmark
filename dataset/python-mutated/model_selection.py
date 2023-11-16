from __future__ import annotations
import itertools

def check_model_selection_order_does_not_matter(model, dataset):
    if False:
        i = 10
        return i + 15
    best_params = []
    permutations = list(itertools.permutations(model.models))
    datasets = itertools.tee(dataset, len(permutations))
    for (permutation, dataset) in zip(permutations, datasets):
        models = [model.clone() for model in permutation]
        clone = model.clone(new_params={'models': models})
        for (x, y) in dataset:
            clone.predict_one(x)
            clone.learn_one(x, y)
        best_params.append(clone.best_model._get_params())
    assert all((params == best_params[0] for params in best_params))