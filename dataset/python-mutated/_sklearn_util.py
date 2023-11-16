def check_fitted(model, func):
    if False:
        print('Hello World!')
    'Check if a model is fitted. Raise error if not.\n\n    Parameters\n    ----------\n    model: model\n        Any scikit-learn model\n\n    func: model\n        Function to check if a model is not trained.\n    '
    if not func(model):
        raise TypeError("Expected a 'fitted' model for conversion")

def check_expected_type(model, expected_type):
    if False:
        for i in range(10):
            print('nop')
    'Check if a model is of the right type. Raise error if not.\n\n    Parameters\n    ----------\n    model: model\n        Any scikit-learn model\n\n    expected_type: Type\n        Expected type of the scikit-learn.\n    '
    if model.__class__.__name__ != expected_type.__name__:
        raise TypeError("Expected model of type '%s' (got %s)" % (expected_type.__name__, model.__class__.__name__))